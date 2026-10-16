#!/usr/bin/env python3
"""
V37 Offline Weight Repacker
===========================
Converts standard GGUF I2_S (type=36) weights from the GGUF file into 
the V37 hardware's expected 2-bit-pair column-interleaved format.

The output .v37pack file is a flat binary that engine.cpp can mmap and
load directly into CMA using V37_MODE_I2_S_PREPACKED — no runtime
conversion needed.

Usage:
    python3 repack_gguf_for_v37.py <input.gguf> <output.v37pack>

Format:
    The .v37pack file contains a simple header followed by repacked weight data:
    - Header: magic(4) + version(4) + n_tensors(4) + reserved(4)
    - For each tensor:
        - name_len(4) + name(name_len) + m(4) + k(4) + scale(4) + data_size(4) + data(data_size)
    - Each tensor's data is in the 4-channel interleaved format expected by CMA.
"""
import struct
import sys
import os
import numpy as np
import time

def read_string(f):
    length = struct.unpack('<Q', f.read(8))[0]
    return f.read(length).decode('utf-8')

def skip_value(f, vtype):
    sizes = {0:1, 1:1, 7:1, 2:2, 3:2, 4:4, 5:4, 6:4, 10:8, 11:8, 12:8}
    if vtype in sizes:
        f.read(sizes[vtype])
    elif vtype == 8:
        length = struct.unpack('<Q', f.read(8))[0]
        f.read(length)
    elif vtype == 9:
        atype = struct.unpack('<I', f.read(4))[0]
        alen = struct.unpack('<Q', f.read(8))[0]
        for _ in range(alen):
            skip_value(f, atype)

def repack_i2s_tensor(raw_data, m, k, w_scale):
    """
    Repack GGUF I2_S raw bytes into V37 hardware format.
    
    Input format (GGUF I2_S, ACT_PARALLEL): QK_I2_S=128, MSB-first
      byte[block*32+group_pos]:
        bits 6-7 = col block*128+group_pos+0*32
        bits 4-5 = col block*128+group_pos+1*32
        bits 2-3 = col block*128+group_pos+2*32
        bits 0-1 = col block*128+group_pos+3*32
      Code: 0=-1, 1=0, 2=+1
    
    Output format (V37 hardware): 4-channel interleaved, 2-bit-pair per byte
      byte[x] = (q[0]<<6)|(q[1]<<4)|(q[2]<<2)|q[3]
      where q[i] is column pld_idx*64 + x + i*16
    """
    padded_k = (k + 255) & ~255
    kb = (k + 63) // 64
    kb = (kb + 3) & ~3  # Pad to multiple of 4 for HW state machine
    
    total_beats = kb  # PLD only for I2_S (no headers needed per V9.5 RTL)
    ch_stride = total_beats * 16
    m_blocks = (m + 3) // 4
    channel_sz = m_blocks * ch_stride
    alloc_sz = 4 * channel_sz
    
    dest = bytearray(alloc_sz)
    
    # Handle negative w_scale
    negate_weights = (w_scale < 0.0)
    
    src = np.frombuffer(raw_data, dtype=np.uint8)
    row_bytes = padded_k // 4
    
    for row in range(m):
        ch_idx = row % 4
        row_in_ch = row // 4
        ch_base = ch_idx * channel_sz + row_in_ch * ch_stride
        
        dest_beat = 0
        pld_idx = 0
        row_byte_offset = row * row_bytes
        
        while pld_idx < kb:
            for b in range(4):
                if pld_idx >= kb:
                    break
                    
                dst_offset = ch_base + dest_beat * 16
                for x in range(16):
                    q = [1, 1, 1, 1]  # Default: 1 maps to 0 in hardware
                    for i in range(4):
                        col = pld_idx * 64 + x + i * 16
                        if col < k:
                            # QK_I2_S=128 MSB-first decode
                            block_idx  = col // 128
                            col_in_blk = col % 128
                            group_idx  = col_in_blk // 32
                            group_pos  = col_in_blk % 32
                            byte_idx   = row_byte_offset + block_idx * 32 + group_pos
                            shift      = 6 - 2 * group_idx
                            code = (int(src[byte_idx]) >> shift) & 3
                            if negate_weights:
                                if code == 0: code = 2    # -1 -> +1
                                elif code == 2: code = 0  # +1 -> -1
                            q[i] = code
                    dest[dst_offset + x] = (q[0] << 6) | (q[1] << 4) | (q[2] << 2) | q[3]
                
                dest_beat += 1
                pld_idx += 1
    
    return bytes(dest), alloc_sz

def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <input.gguf> <output.v37pack>")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    print(f"[V37-REPACK] Reading GGUF: {input_path}")
    
    # Target I2_S tensors to repack
    target_names = set()
    for l in range(30):
        for suffix in ["attn_q.weight", "attn_k.weight", "attn_v.weight", 
                       "attn_output.weight", "ffn_gate.weight", "ffn_up.weight", 
                       "ffn_down.weight"]:
            target_names.add(f"blk.{l}.{suffix}")
    target_names.add("output.weight")
    
    with open(input_path, 'rb') as f:
        magic = f.read(4)
        assert magic == b'GGUF', f"Not a GGUF file: {magic}"
        version = struct.unpack('<I', f.read(4))[0]
        n_tensors = struct.unpack('<Q', f.read(8))[0]
        n_kv = struct.unpack('<Q', f.read(8))[0]
        
        alignment = 32
        for i in range(n_kv):
            key = read_string(f)
            vtype = struct.unpack('<I', f.read(4))[0]
            if key == "general.alignment" and vtype == 4:
                alignment = struct.unpack('<I', f.read(4))[0]
            else:
                skip_value(f, vtype)
        
        tensors = []
        for i in range(n_tensors):
            name = read_string(f)
            n_dims = struct.unpack('<I', f.read(4))[0]
            dims = [struct.unpack('<Q', f.read(8))[0] for _ in range(n_dims)]
            ttype = struct.unpack('<I', f.read(4))[0]
            offset = struct.unpack('<Q', f.read(8))[0]
            tensors.append((name, ttype, dims, offset))
        
        data_offset = (f.tell() + alignment - 1) & ~(alignment - 1)
        
        # Process each I2_S tensor
        repacked_tensors = []
        total_start = time.time()
        
        for name, ttype, dims, offset in tensors:
            if ttype != 36 or name not in target_names:
                continue
            
            k = dims[0]  # ne[0] = input dim
            m = dims[1] if len(dims) > 1 else 1  # ne[1] = output dim
            elements = k * m
            padded_k = (k + 255) & ~255
            packed_bytes = elements // 4
            
            # Read weight data
            f.seek(data_offset + offset)
            raw_data = f.read(packed_bytes)
            
            # Read scale (embedded after packed data in I2_S format)
            scale_data = f.read(4)
            w_scale = struct.unpack('<f', scale_data)[0]
            
            print(f"  [{name}] m={m}, k={k}, scale={w_scale:.6f}, repacking...", end="", flush=True)
            t0 = time.time()
            
            repacked_data, alloc_sz = repack_i2s_tensor(raw_data, m, k, w_scale)
            
            elapsed = time.time() - t0
            print(f" done ({elapsed:.1f}s, {alloc_sz/1024/1024:.1f}MB)")
            
            repacked_tensors.append({
                'name': name,
                'm': m,
                'k': k,
                'scale': abs(w_scale),  # Store absolute scale (negation already baked into encoding)
                'data': repacked_data,
            })
    
    total_elapsed = time.time() - total_start
    
    # Write output .v37pack file
    print(f"\n[V37-REPACK] Writing {len(repacked_tensors)} tensors to {output_path}...")
    
    with open(output_path, 'wb') as f:
        # Header
        f.write(b'V37P')  # magic
        f.write(struct.pack('<I', 1))  # version
        f.write(struct.pack('<I', len(repacked_tensors)))
        f.write(struct.pack('<I', 0))  # reserved
        
        for t in repacked_tensors:
            name_bytes = t['name'].encode('utf-8')
            f.write(struct.pack('<I', len(name_bytes)))
            f.write(name_bytes)
            f.write(struct.pack('<I', t['m']))
            f.write(struct.pack('<I', t['k']))
            f.write(struct.pack('<f', t['scale']))
            f.write(struct.pack('<I', len(t['data'])))
            f.write(t['data'])
    
    output_size = os.path.getsize(output_path)
    print(f"[V37-REPACK] Done! Output: {output_size/1024/1024:.1f}MB, Time: {total_elapsed:.1f}s")

if __name__ == '__main__':
    main()
