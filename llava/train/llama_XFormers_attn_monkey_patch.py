from typing import Optional, Tuple
import warnings

import torch

import transformers
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv

from xformers.ops import memory_efficient_attention, AttentionMask


def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    if output_attentions:
        warnings.warn("Output attentions not supported for this attention implementation.")

    bsz, q_len, _ = hidden_states.size()

    # Project
    query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim)
    key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim)
    value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim)

    # Transpose for attention: [B, H, S, D]
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)
        kv_seq_len = key_states.shape[2]

    # Rotary embedding
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    past_key_value = (key_states, value_states) if use_cache else None

    # Repeat k/v if needed
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    # Prepare input: [B * H, S, D]
    q = query_states.transpose(1, 2).reshape(bsz * self.num_heads, q_len, self.head_dim)
    k = key_states.transpose(1, 2).reshape(bsz * self.num_heads, -1, self.head_dim)
    v = value_states.transpose(1, 2).reshape(bsz * self.num_heads, -1, self.head_dim)

    # Attention mask
    attn_mask = None
    if attention_mask is not None:
        attention_mask = attention_mask[:, None, None, :]  # [B, 1, 1, S]
        attn_mask = AttentionMask.from_bool(attention_mask.to(torch.bool))

    # XFormers attention
    attn_output = memory_efficient_attention(q, k, v, attn_mask=attn_mask, causal=True)

    # Reshape and output projection
    attn_output = attn_output.view(bsz, self.num_heads, q_len, self.head_dim).transpose(1, 2)
    attn_output = attn_output.reshape(bsz, q_len, -1)

    return self.o_proj(attn_output), None, past_key_value


# Disable the transformation of the attention mask in LlamaModel as the flash attention
# requires the attention mask to be the same as the key_padding_mask
def _prepare_decoder_attention_mask(
        self, attention_mask, input_shape, inputs_embeds, past_key_values_length
):
    # [bsz, seq_len]
    return attention_mask


def replace_llama_attn_with_flash_attn():
    cuda_major, cuda_minor = torch.cuda.get_device_capability()
    if cuda_major < 8:
        warnings.warn(
            "Flash attention is only supported on A100 or H100 GPU during training due to head dim > 64 backward."
            "ref: https://github.com/HazyResearch/flash-attention/issues/190#issuecomment-1523359593"
        )
    transformers.models.llama.modeling_llama.LlamaModel._prepare_decoder_attention_mask = (
        _prepare_decoder_attention_mask
    )
    transformers.models.llama.modeling_llama.LlamaAttention.forward = forward
