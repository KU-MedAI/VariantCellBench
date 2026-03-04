import torch
from torch import nn
import pandas as pd
from pretrainmodels.performer import PerformerModule
from pretrainmodels.transformer import pytorchTransformerModule
from pretrainmodels.mae_autobin import AutoDiscretizationEmbedding2


class MaeAutobin(nn.Module):
    def __init__(self, *,
        num_tokens, max_seq_len, embed_dim, decoder_embed_dim,
        bin_alpha=1.0, bin_num=100, pad_token_id=103, mask_token_id=102,
        cond_in_dim=1280,                              # 변이 임베딩 차원
        gene_index_tsv="/home/tech/variantseq/foundation/scFoundation/model/OS_scRNA_gene_index.19264.tsv",
        embedding_cache=None,                           # {(gene, variant): np.ndarray or Tensor}
        model_config,
        config
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.num_tokens  = num_tokens
        self.pad_token_id = pad_token_id
        self.mask_token_id = mask_token_id

        # (1) 임베딩
        self.token_emb = AutoDiscretizationEmbedding2(
            embed_dim, max_seq_len, bin_num=bin_num, bin_alpha=bin_alpha,
            pad_token_id=pad_token_id, mask_token_id=mask_token_id
        )
        self.pos_emb = nn.Embedding(max_seq_len + 1, embed_dim)

        # (2) 변이 임베딩 축소 MLP
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_in_dim, embed_dim, bias=True),
        )

        # (3) gene index 로드 < - 이거 기준으로 variant 매핑할겁니다.
        # scFoundation 기준 유전자 리스트
        gi = pd.read_csv(gene_index_tsv, sep="\t")
        self.gene2idx = {str(g): int(i) for g, i in zip(gi["gene_name"], gi["index"])}

        self.embedding_cache = embedding_cache or {}


        # ----- (4) Encoder/Decoder/Head (기존 구성 유지하되 헤드/차원 일관성 체크) -----
        enc_cfg = model_config['encoder']
        dec_cfg = model_config['decoder']

        enc_dim   = enc_cfg['hidden_dim']
        enc_heads = enc_cfg.get('heads')
        if enc_heads is None:
            enc_heads = _pick_num_heads(enc_dim, max_head_dim=64)
        _validate_heads("encoder", enc_dim, enc_heads)

        dec_dim   = dec_cfg['hidden_dim']
        dec_heads = dec_cfg.get('heads')
        if dec_heads is None:
            dec_heads = _pick_num_heads(dec_dim, max_head_dim=64)
        _validate_heads("decoder", dec_dim, dec_heads)

        # Performer용 dim_head 일관화: 제공 안되면 자동 유도
        dec_dim_head = dec_cfg.get('dim_head')
        if dec_dim_head is None:
            # 가장 흔한 선택: dec_dim // dec_heads
            dec_dim_head = dec_dim // dec_heads
        # Performer 구현이 (heads * dim_head == dim) 전제를 가진다면 체크
        assert dec_heads * dec_dim_head == dec_dim, \
            f"[decoder] heads({dec_heads}) * dim_head({dec_dim_head}) != hidden_dim({dec_dim})"
        

        # (4) Encoder/Decoder/Head (기존 구성 유지)
        # Encoder
        self.encoder = pytorchTransformerModule(
            max_seq_len,
            dim=enc_dim,
            depth=enc_cfg['depth'],
            heads=enc_heads,
        )

        # Decoder (Performer)
        self.decoder = PerformerModule(
            max_seq_len,
            dim=dec_dim,
            depth=dec_cfg['depth'],
            heads=dec_heads,
            dim_head=dec_dim_head,
            ff_dropout=dec_cfg.get('ff_dropout', 0.0),
            attn_dropout=dec_cfg.get('attn_dropout', 0.0),
        )
        
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim)
        self.norm = nn.LayerNorm(decoder_embed_dim)
        self.to_final = nn.Linear(decoder_embed_dim, 1)
        self.config = config

    # === 헬퍼들 ===
    def _get_cached_embedding(self, gene, var_key):
        arr = self.embedding_cache.get((gene, var_key), None)
        if arr is None: return None
        t = torch.as_tensor(arr) if not torch.is_tensor(arr) else arr
        if t.dim() == 2: t = t.unsqueeze(0)      # (L, Dv) -> (1, L, Dv)
        return t

    @staticmethod
    def _pad_to_length(t: torch.Tensor, target_L: int):
        B, L, D = t.shape
        if L >= target_L: return t
        return torch.cat([t, t.new_zeros((B, target_L - L, D))], dim=1)

    # === forward ===
    def forward(self, x, padding_label, encoder_position_gene_ids, encoder_labels,
                decoder_data, mask_gene_name, mask_labels, decoder_position_gene_ids,
                decoder_data_padding_labels, data, output_attentions=False, **kwargs):
        variant_representation = self.config.get("variant_representation", 'ALT') # 'ALT' / 'DIFF'
        compression = self.config.get("compression", 'position_embedding') # 'full_sequence_average' / 'position_embedding'
        B, Kenc = x.shape[0], x.shape[1]
        device, f32 = x.device, torch.float32

        # 1) 토큰+포지션 임베딩
        x = self.token_emb(torch.unsqueeze(x, 2), output_weight=0)      # [B, Kenc, E]
        if output_attentions: x.requires_grad_()
        pos = self.pos_emb(encoder_position_gene_ids)                    # [B, Kenc, E]
        x = x + pos

        # 2) 변이 주입 (batch["variant"] = List[List["GENE~ALT"]])
        cond_batch = kwargs.get("variant", None) or kwargs.get("condition", None)
        if cond_batch is not None:
            for b in range(B):
                specs = cond_batch[b]
                if not specs: continue
                # 유전자별 누적 벡터
                acc = {}
                for spec in specs:
                    if not isinstance(spec, str) or "~" not in spec: continue
                    gene_, var_ = spec.split("~", 1)
                    # 유전자 index 가져옵니다. (ex. TP53 -> n)
                    gidx = int(self.gene2idx.get(gene_, -1))
                    if gidx < 0: continue

                    # ALT와 REF 시퀀스 임베딩을 캐시파일에서 로드해옵니다. 캐시파일은 데이터 로드 시 불러옵니다.
                        # 전체 서열 임베딩을 사용하는 경우. REF와 ALT를 각각 가져옴
                    if compression == 'full_sequence_average':
                        ref = get_cached_embedding(gene_, "REF", data.embedding_cache)
                        alt = get_cached_embedding(gene_, var_, data.embedding_cache)
                    # 변이 발생 위치의 임베딩을 사용하는 경우.
                        # 설정한 variant_representation에 따라 ALT or DIFF 임베딩을 가져옴
                    elif compression == 'position_embedding':
                        ref = get_cached_embedding(gene_, var_, data.embedding_cache, variant_representation)
                        alt = get_cached_embedding(gene_, var_, data.embedding_cache, variant_representation)
                    else:
                        pass
                    
                    if ref is None or alt is None: continue
                    # 불필요한 변환 방지
                    if not isinstance(ref, torch.Tensor):
                        ref = torch.tensor(ref, dtype=torch.float32, device=device)
                        alt = torch.tensor(alt, dtype=torch.float32, device=device)

                    else:
                        ref = ref.to(dtype=torch.float32, device=device)
                        alt = alt.to(dtype=torch.float32, device=device)


                    if compression == 'full_sequence_average':
                        L = max(ref.shape[1], alt.shape[1])
                        ref = self._pad_to_length(torch.tensor(ref), L)
                        alt = self._pad_to_length(torch.tensor(alt), L)
                        if variant_representation == 'DIFF':
                            delta = (ref - alt).mean(dim=1).squeeze(0)           # [Dv]
                        elif variant_representation == 'ALT':
                            delta = alt.mean(dim=1).squeeze(0)
                    elif compression == 'position_embedding':
                        delta = alt

                    vecE  = self.cond_mlp(delta).to(x.dtype)             # [E]
                    acc[gidx] = acc.get(gidx, 0) + vecE

                if acc:
                    pos_ids_b = encoder_position_gene_ids[b]             # [Kenc]
                    for gidx, v in acc.items():
                        m = (pos_ids_b == gidx)
                        if m.any():
                            x[b, m] = x[b, m] + v

        # 3) Encoder
        x = self.encoder(x, padding_mask=padding_label)

        # 4) Decoder 입력 준비 및 주입
        dec = self.token_emb(torch.unsqueeze(decoder_data, 2))
        posd = self.pos_emb(decoder_position_gene_ids)
        if mask_gene_name:
            raise NotImplementedError("mask_gene_name")
        batch_idx, gen_idx = (encoder_labels == True).nonzero(as_tuple=True)
        dec[batch_idx, gen_idx] = x[~padding_label].to(dec.dtype)
        dec = dec + posd
        dec = self.decoder_embed(dec)

        # 5) Decoder
        x = self.decoder(dec, padding_mask=decoder_data_padding_labels)
        x = self.norm(x)
        x = self.to_final(x)
        return x.squeeze(2)                     # [B, N]


def _pick_num_heads(d_model: int, max_head_dim: int = 64) -> int:
    """d_model을 나누는 값 중 head_dim <= max_head_dim인 가장 큰 head 수 반환"""
    cands = [h for h in range(1, d_model + 1) if d_model % h == 0 and (d_model // h) <= max_head_dim]
    return max(cands) if cands else 1

def _validate_heads(name: str, d_model: int, heads: int):
    assert d_model % heads == 0, f"[{name}] d_model({d_model}) % heads({heads}) != 0"