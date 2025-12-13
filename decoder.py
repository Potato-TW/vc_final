import numpy as np
import math
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, BinaryIO, Optional
from collections import defaultdict
from codec import Zigzag

# --- Data Structures using Dataclasses ---

@dataclass
class HuffmanTable:
    table_class: int
    dest_id: int
    counts: List[int]
    # Mapping: (code_value, bit_length) -> symbol
    huff_data: Dict[Tuple[int, int], int] = field(default_factory=dict)

@dataclass
class QuantizationTable:
    precision: int
    dest_id: int
    table: np.ndarray 

@dataclass
class FrameComponent:
    identifier: int
    sampling_factor: int
    h_sampling_factor: int
    v_sampling_factor: int
    quant_table_dest: int

@dataclass
class StartOfFrame:
    precision: int
    num_lines: int
    samples_per_line: int
    components: List[FrameComponent] = field(default_factory=list)

@dataclass
class ScanComponent:
    selector: int
    dc_table: int
    ac_table: int

@dataclass
class StartOfScan:
    components: List[ScanComponent] = field(default_factory=list)
    spectral_selection_range: Tuple[int, int] = (0, 63)
    successive_approximation: int = 0

# --- Helper Classes ---

class BitStream:
    """Handles efficient bit-level reading from a byte buffer."""
    def __init__(self, data: bytearray):
        self.data = data
        self.byte_pos = 0
        self.bit_pos = 0  # 0 to 7, where 0 is MSB
        self.data_len = len(data)

    def get_bit(self) -> int:
        if self.byte_pos >= self.data_len:
            return 0
        
        # Extract bit at current position (Big Endian)
        val = (self.data[self.byte_pos] >> (7 - self.bit_pos)) & 1
        
        self.bit_pos += 1
        if self.bit_pos == 8:
            self.bit_pos = 0
            self.byte_pos += 1
        return val

    def get_bits(self, num_bits: int) -> int:
        val = 0
        for _ in range(num_bits):
            val = (val << 1) | self.get_bit()
        return val

# --- Main Decoder ---

class BaselineJPEGDecoder:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.huffman_tables = {}
        self.quantization_tables = {}
        self.sos: Optional[StartOfScan] = None
        self.sof: Optional[StartOfFrame] = None
        self.zigzag = Zigzag.get_zigzag(8)
        
        # Precompute IDCT transformation matrix (8x8)
        self.idct_base_matrix = self._create_idct_matrix()

    def decode(self) -> np.ndarray:
        with open(self.filepath, 'rb') as f:
            data = f.read()

        pos = 0
        while pos < len(data):
            if data[pos] != 0xFF:
                pos += 1
                continue
            
            marker = data[pos:pos+2]
            pos += 2
            
            marker_val = int.from_bytes(marker, 'big')

            if marker_val == 0xFFD8: # SOI
                continue
            elif marker_val == 0xFFD9: # EOI
                break
            
            length = int.from_bytes(data[pos:pos+2], 'big')
            segment_data = data[pos+2 : pos+length]
            
            if marker_val == 0xFFC0: # SOF0
                self.sof = self._parse_sof(segment_data)
            elif marker_val == 0xFFDB: # DQT
                self._parse_dqt(segment_data)
            elif marker_val == 0xFFC4: # DHT
                self._parse_dht(segment_data)
            elif marker_val == 0xFFDA: # SOS
                self.sos = self._parse_sos(segment_data)
                pos += length 
                return self._process_scan_data(data[pos:])
            
            pos += length

        raise ValueError("Image data incomplete or no SOS marker found.")

    def _process_scan_data(self, raw_data: bytes) -> np.ndarray:
        # 1. Remove Byte Stuffing (0xFF00 -> 0xFF)
        scan_data = bytearray()
        i = 0
        while i < len(raw_data):
            b = raw_data[i]
            if b == 0xFF:
                if i + 1 < len(raw_data) and raw_data[i+1] == 0x00:
                    scan_data.append(0xFF)
                    i += 2
                    continue
            scan_data.append(b)
            i += 1

        bit_stream = BitStream(scan_data)
        
        assert self.sof is not None
        width = self.sof.samples_per_line
        height = self.sof.num_lines
        
        max_h = max(c.h_sampling_factor for c in self.sof.components)
        max_v = max(c.v_sampling_factor for c in self.sof.components)
        mcu_width = 8 * max_h
        mcu_height = 8 * max_v
        
        mcus_x = math.ceil(width / mcu_width)
        mcus_y = math.ceil(height / mcu_height)
        
        ycbcr_buffer = np.zeros((height, width, 3), dtype=np.float32)
        dc_preds = [0] * len(self.sos.components)
        
        for mcu_y in range(mcus_y):
            for mcu_x in range(mcus_x):
                for i, scan_comp in enumerate(self.sos.components):
                    frame_comp = next(c for c in self.sof.components if c.identifier == scan_comp.selector)
                    
                    qt = self.quantization_tables[frame_comp.quant_table_dest]
                    dct_dc = self.huffman_tables[(scan_comp.dc_table, 0)]
                    dct_ac = self.huffman_tables[(scan_comp.ac_table, 1)]
                    
                    for v_block in range(frame_comp.v_sampling_factor):
                        for h_block in range(frame_comp.h_sampling_factor):
                            
                            # 1. Huffman Decode
                            block = self._decode_block(bit_stream, dct_dc, dct_ac, dc_preds, i)
                            
                            # 2. Dequantize
                            block = block * qt.table
                            
                            # 3. IDCT (Vectorized)
                            # FIX: Correct Order is C * Block * C.T
                            # C (spatial, freq) @ Block (freq, freq) @ C.T (freq, spatial) -> (spatial, spatial)
                            block = self.idct_base_matrix @ block @ self.idct_base_matrix.T
                            block = block + 128
                            
                            # 4. Place into buffer
                            base_y = (mcu_y * mcu_height) + (v_block * 8)
                            base_x = (mcu_x * mcu_width) + (h_block * 8)
                            
                            h_scale = max_h // frame_comp.h_sampling_factor
                            v_scale = max_v // frame_comp.v_sampling_factor
                            
                            if h_scale == 1 and v_scale == 1:
                                y_end = min(base_y + 8, height)
                                x_end = min(base_x + 8, width)
                                ycbcr_buffer[base_y:y_end, base_x:x_end, i] = block[:y_end-base_y, :x_end-base_x]
                            else:
                                expanded = np.repeat(np.repeat(block, v_scale, axis=0), h_scale, axis=1)
                                
                                real_y = (mcu_y * mcu_height) + (v_block * 8 * v_scale)
                                real_x = (mcu_x * mcu_width) + (h_block * 8 * h_scale)
                                
                                y_end = min(real_y + expanded.shape[0], height)
                                x_end = min(real_x + expanded.shape[1], width)
                                
                                ycbcr_buffer[real_y:y_end, real_x:x_end, i] = expanded[:y_end-real_y, :x_end-real_x]

        return self._ycbcr_to_rgb_vectorized(ycbcr_buffer), self._extract_y(ycbcr_buffer)

    def _extract_y(self, img_ycbcr: np.ndarray) -> np.ndarray:
        Y = img_ycbcr[:, :, 0]
        return np.clip(Y, 0, 255).astype(np.uint8)


    def _decode_block(self, bits: BitStream, dc_table: Dict, ac_table: Dict, preds: List[int], comp_idx: int) -> np.ndarray:
        s, _ = self._huffman_decode(bits, dc_table)
        diff = self._get_value(bits, s) if s > 0 else 0
        preds[comp_idx] += diff
        
        coeffs = np.zeros(64, dtype=int)
        coeffs[0] = preds[comp_idx]
        
        k = 1
        while k < 64:
            s, _ = self._huffman_decode(bits, ac_table)
            r = s >> 4
            s = s & 0x0F
            
            if s == 0:
                if r == 15: k += 16; continue
                else: break
            
            k += r
            if k >= 64: break
            
            coeffs[k] = self._get_value(bits, s)
            k += 1
            
        block = np.zeros((8, 8), dtype=float)
        for idx, val in enumerate(coeffs):
            if val != 0:
                r, c = self.zigzag[idx]
                block[r, c] = val
        return block

    def _huffman_decode(self, bits: BitStream, huff_table: Dict) -> Tuple[int, int]:
        code = 0
        length = 0
        while True:
            bit = bits.get_bit()
            code = (code << 1) | bit
            length += 1
            
            if (code, length) in huff_table:
                return huff_table[(code, length)], length
                
            if length > 16:
                raise ValueError("Huffman decode error")

    def _get_value(self, bits: BitStream, length: int) -> int:
        val = bits.get_bits(length)
        if val < (1 << (length - 1)):
            return val - ((1 << length) - 1)
        return val

    def _create_idct_matrix(self) -> np.ndarray:
        # Standard IDCT Matrix C
        # Rows = Spatial (x), Cols = Frequency (u)
        C = np.zeros((8, 8))
        for u in range(8):
            for x in range(8):
                alpha = 1 / math.sqrt(2) if u == 0 else 1
                C[x, u] = 0.5 * alpha * math.cos(((2 * x + 1) * u * math.pi) / 16)
        return C

    def _ycbcr_to_rgb_vectorized(self, img_ycbcr: np.ndarray) -> np.ndarray:
        img_rgb = np.zeros_like(img_ycbcr, dtype=np.uint8)
        
        Y = img_ycbcr[:, :, 0]
        Cb = img_ycbcr[:, :, 1] - 128
        Cr = img_ycbcr[:, :, 2] - 128
        
        R = Y + 1.402 * Cr
        G = Y - 0.344136 * Cb - 0.714136 * Cr
        B = Y + 1.772 * Cb
        
        # Explicit casting to uint8 after clipping to fix the crash
        img_rgb[:, :, 0] = np.clip(R, 0, 255).astype(np.uint8)
        img_rgb[:, :, 1] = np.clip(G, 0, 255).astype(np.uint8)
        img_rgb[:, :, 2] = np.clip(B, 0, 255).astype(np.uint8)
        
        return img_rgb

    # --- Header Parsers ---

    def _parse_sof(self, data: bytes) -> StartOfFrame:
        p, h, w, nf = data[0], int.from_bytes(data[1:3], 'big'), int.from_bytes(data[3:5], 'big'), data[5]
        comps = []
        idx = 6
        for _ in range(nf):
            id_ = data[idx]
            samp = data[idx+1]
            qtd = data[idx+2]
            comps.append(FrameComponent(id_, samp, samp >> 4, samp & 0x0F, qtd))
            idx += 3
        return StartOfFrame(p, h, w, comps)

    def _parse_dqt(self, data: bytes):
        idx = 0
        while idx < len(data):
            info = data[idx]
            precision = info >> 4
            dest_id = info & 0x0F
            idx += 1
            
            # 修正：根據 precision 決定讀取 1 或 2 bytes
            element_bytes = 1 if precision == 0 else 2
            
            table_vals = []
            for i in range(64):
                # 修正：正確讀取 element_bytes 長度的數值
                val = int.from_bytes(data[idx : idx + element_bytes], 'big')
                table_vals.append(val)
                idx += element_bytes
            
            q_mat = np.zeros((8, 8), dtype=float)
            for i, val in enumerate(table_vals):
                r, c = self.zigzag[i]
                q_mat[r, c] = val
                
            self.quantization_tables[dest_id] = QuantizationTable(precision, dest_id, q_mat)

    def _parse_dht(self, data: bytes):
        idx = 0
        while idx < len(data):
            info = data[idx]
            tc = info >> 4
            th = info & 0x0F
            idx += 1
            
            counts = list(data[idx : idx+16])
            idx += 16
            
            huff_data = {}
            code = 0
            for length, count in enumerate(counts, 1):
                for _ in range(count):
                    val = data[idx]
                    huff_data[(code, length)] = val
                    code += 1
                    idx += 1
                code <<= 1
                
            self.huffman_tables[(th, tc)] = huff_data

    def _parse_sos(self, data: bytes) -> StartOfScan:
        ns = data[0]
        idx = 1
        comps = []
        for _ in range(ns):
            sel = data[idx]
            tbl = data[idx+1]
            comps.append(ScanComponent(sel, tbl >> 4, tbl & 0x0F))
            idx += 2
        
        ss, se = data[idx], data[idx+1]
        approx = data[idx+2]
        return StartOfScan(comps, (ss, se), approx)