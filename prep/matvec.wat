;; matvec.wat: the one hot loop of llm.js (knobs x numbers), in WebAssembly with 128-bit SIMD (four 32-bit floats
;; per instruction). llm.js keeps every model's knobs inside this module's memory so these functions can read them
;; directly; anything else falls back to the plain JavaScript loop, which does the same sums.
;; Assemble with prep/assemble_wasm.mjs (wabt's wat2wasm); the bytes are pasted into llm.js as base64.
(module
  (import "env" "mem" (memory 1 65536))

  ;; out[r] = sum over c of W[r*cols + c] * x[c], for r < rows. Pointers are byte offsets; cols is a multiple of 16.
  (func (export "matvec") (param $w i32) (param $rows i32) (param $cols i32) (param $x i32) (param $out i32)
    (local $r i32) (local $c i32) (local $row i32) (local $end i32)
    (local $a0 v128) (local $a1 v128) (local $a2 v128) (local $a3 v128)
    (local.set $row (local.get $w))
    (block $done
      (loop $rows_loop
        (br_if $done (i32.ge_u (local.get $r) (local.get $rows)))
        (local.set $a0 (v128.const f32x4 0 0 0 0)) (local.set $a1 (v128.const f32x4 0 0 0 0))
        (local.set $a2 (v128.const f32x4 0 0 0 0)) (local.set $a3 (v128.const f32x4 0 0 0 0))
        (local.set $c (i32.const 0))
        (local.set $end (i32.shl (local.get $cols) (i32.const 2)))
        (loop $cols_loop
          (local.set $a0 (f32x4.add (local.get $a0) (f32x4.mul (v128.load (i32.add (local.get $row) (local.get $c))) (v128.load (i32.add (local.get $x) (local.get $c))))))
          (local.set $a1 (f32x4.add (local.get $a1) (f32x4.mul (v128.load offset=16 (i32.add (local.get $row) (local.get $c))) (v128.load offset=16 (i32.add (local.get $x) (local.get $c))))))
          (local.set $a2 (f32x4.add (local.get $a2) (f32x4.mul (v128.load offset=32 (i32.add (local.get $row) (local.get $c))) (v128.load offset=32 (i32.add (local.get $x) (local.get $c))))))
          (local.set $a3 (f32x4.add (local.get $a3) (f32x4.mul (v128.load offset=48 (i32.add (local.get $row) (local.get $c))) (v128.load offset=48 (i32.add (local.get $x) (local.get $c))))))
          (local.set $c (i32.add (local.get $c) (i32.const 64)))
          (br_if $cols_loop (i32.lt_u (local.get $c) (local.get $end))))
        (local.set $a0 (f32x4.add (f32x4.add (local.get $a0) (local.get $a1)) (f32x4.add (local.get $a2) (local.get $a3))))
        (f32.store (i32.add (local.get $out) (i32.shl (local.get $r) (i32.const 2)))
          (f32.add (f32.add (f32x4.extract_lane 0 (local.get $a0)) (f32x4.extract_lane 1 (local.get $a0)))
                   (f32.add (f32x4.extract_lane 2 (local.get $a0)) (f32x4.extract_lane 3 (local.get $a0)))))
        (local.set $row (i32.add (local.get $row) (local.get $end)))
        (local.set $r (i32.add (local.get $r) (i32.const 1)))
        (br $rows_loop))))

  ;; The same for four tokens at once (reading a prompt): each row of knobs is loaded once and used four times.
  ;; x0..x3 and o0..o3 are the four tokens' inputs and outputs.
  (func (export "matmat4") (param $w i32) (param $rows i32) (param $cols i32)
        (param $x0 i32) (param $x1 i32) (param $x2 i32) (param $x3 i32)
        (param $o0 i32) (param $o1 i32) (param $o2 i32) (param $o3 i32)
    (local $r i32) (local $c i32) (local $row i32) (local $end i32) (local $wv v128) (local $ro i32)
    (local $a0 v128) (local $a1 v128) (local $a2 v128) (local $a3 v128)
    (local.set $row (local.get $w))
    (local.set $end (i32.shl (local.get $cols) (i32.const 2)))
    (block $done
      (loop $rows_loop
        (br_if $done (i32.ge_u (local.get $r) (local.get $rows)))
        (local.set $a0 (v128.const f32x4 0 0 0 0)) (local.set $a1 (v128.const f32x4 0 0 0 0))
        (local.set $a2 (v128.const f32x4 0 0 0 0)) (local.set $a3 (v128.const f32x4 0 0 0 0))
        (local.set $c (i32.const 0))
        (loop $cols_loop
          (local.set $wv (v128.load (i32.add (local.get $row) (local.get $c))))
          (local.set $a0 (f32x4.add (local.get $a0) (f32x4.mul (local.get $wv) (v128.load (i32.add (local.get $x0) (local.get $c))))))
          (local.set $a1 (f32x4.add (local.get $a1) (f32x4.mul (local.get $wv) (v128.load (i32.add (local.get $x1) (local.get $c))))))
          (local.set $a2 (f32x4.add (local.get $a2) (f32x4.mul (local.get $wv) (v128.load (i32.add (local.get $x2) (local.get $c))))))
          (local.set $a3 (f32x4.add (local.get $a3) (f32x4.mul (local.get $wv) (v128.load (i32.add (local.get $x3) (local.get $c))))))
          (local.set $c (i32.add (local.get $c) (i32.const 16)))
          (br_if $cols_loop (i32.lt_u (local.get $c) (local.get $end))))
        (local.set $ro (i32.shl (local.get $r) (i32.const 2)))
        (f32.store (i32.add (local.get $o0) (local.get $ro)) (f32.add (f32.add (f32x4.extract_lane 0 (local.get $a0)) (f32x4.extract_lane 1 (local.get $a0))) (f32.add (f32x4.extract_lane 2 (local.get $a0)) (f32x4.extract_lane 3 (local.get $a0)))))
        (f32.store (i32.add (local.get $o1) (local.get $ro)) (f32.add (f32.add (f32x4.extract_lane 0 (local.get $a1)) (f32x4.extract_lane 1 (local.get $a1))) (f32.add (f32x4.extract_lane 2 (local.get $a1)) (f32x4.extract_lane 3 (local.get $a1)))))
        (f32.store (i32.add (local.get $o2) (local.get $ro)) (f32.add (f32.add (f32x4.extract_lane 0 (local.get $a2)) (f32x4.extract_lane 1 (local.get $a2))) (f32.add (f32x4.extract_lane 2 (local.get $a2)) (f32x4.extract_lane 3 (local.get $a2)))))
        (f32.store (i32.add (local.get $o3) (local.get $ro)) (f32.add (f32.add (f32x4.extract_lane 0 (local.get $a3)) (f32x4.extract_lane 1 (local.get $a3))) (f32.add (f32x4.extract_lane 2 (local.get $a3)) (f32x4.extract_lane 3 (local.get $a3)))))
        (local.set $row (i32.add (local.get $row) (local.get $end)))
        (local.set $r (i32.add (local.get $r) (i32.const 1)))
        (br $rows_loop))))
)
