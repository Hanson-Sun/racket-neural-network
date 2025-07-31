#lang typed/racket
(require rackunit)
(require/typed rackunit
  [check-equal? (-> Any Any Void)]
  [check-= (-> Number Number Number Void)])
(require racket/random)



(provide matrix
         matrix-cols
         matrix-rows
         matrix-data)
(struct matrix ([cols : Integer]
                [rows : Integer]
                [data : (Vectorof Number)])
  #:transparent)

(provide make-matrix-fill)
(define (make-matrix-fill [cols : Integer]
                          [rows : Integer]
                          [fill : Integer])
  (matrix cols rows (make-vector (* cols rows) fill)))

(provide make-matrix-random)
(define (make-matrix-random [cols : Integer]
                            [rows : Integer]
                            [min : Number -1]
                            [max : Number 1])
  (define total (* cols rows))
  (define data
    (build-vector total (λ: ([i : Integer])
                        (+ min (* (random) (- max min))))))
  (matrix cols rows data))

(provide make-matrix)
(define (make-matrix [cols : Integer]
                     [rows : Integer]
                     [data : (Vectorof Number)])
  (matrix cols rows data))

;; get data vector index
(provide matrix-index)
(define (matrix-index [x : Integer]
                      [y : Integer]
                      [cols : Integer])
  (+ (* y cols) x))

;; get cell
(provide matrix-get)
(define (matrix-get [mat : matrix]
                    [x : Integer]
                    [y : Integer])
  (vector-ref (matrix-data mat) (matrix-index x y (matrix-cols mat))))

;; elementwise matrix op
(provide matrix-elem)
(: matrix-elem (matrix matrix (Number Number -> Number) -> matrix))
(define (matrix-elem [m1 : matrix]
                     [m2 : matrix]
                     [func : (Number Number -> Number)])
  (define cols (matrix-cols m1))
  (define rows (matrix-rows m1))
  (define v1 (matrix-data m1))
  (define v2 (matrix-data m2))
  (unless (and (= cols (matrix-cols m2)) (= rows (matrix-rows m2)))
      (error "matrix dims must match for add"))

  (define result
    (build-vector (* cols rows)
                  (λ: ([i : Integer])
                    (func (vector-ref v1 i) (vector-ref v2 i)))))

  (make-matrix cols rows result))

;; mat scalar multiplication
(provide matrix-sm)
(: matrix-sm (matrix Number -> matrix))
(define (matrix-sm [m : matrix]
                   [s : Number])
  (define cols (matrix-cols m))
  (define rows (matrix-rows m))
  (define v (matrix-data m))
  (make-matrix cols rows (build-vector (* cols rows)
                  (λ: ([i : Integer])
                    (* s (vector-ref v i))))))


;; transpose
(provide matrix-transpose)
(: matrix-transpose (matrix -> matrix))
(define (matrix-transpose [m : matrix])
  (define cols (matrix-cols m))
  (define rows (matrix-rows m))
  (define v (matrix-data m))
  (make-matrix rows cols
               (build-vector (* cols rows)
                                       (λ: ([i : Integer])
                                          (vector-ref v (+ (* (modulo i rows) cols) (floor (/ i rows))))))))

(define mat1 (make-matrix 2 3 #(1 2 3 4 5 6)))
(define trans1 (matrix-transpose mat1))
(check-equal? (matrix-cols trans1) 3)
(check-equal? (matrix-rows trans1) 2)
(check-equal? (matrix-data trans1) #(1 3 5 2 4 6))
(define mat2 (make-matrix 2 2 #(1 2 3 4)))
(define trans2 (matrix-transpose mat2))
(check-equal? (matrix-cols trans2) 2)
(check-equal? (matrix-rows trans2) 2)
(check-equal? (matrix-data trans2) #(1 3 2 4))

;; TODO: strassen
;; (define (matrix-strassen [m1 : matrix]
;;                          [m2 : matrix])
;;   (...)




;; matmul (tears). Its the canonical matmul: a* (m1 * m2) + b)
(provide matrix-matmul)
(: matrix-matmul
   (case->
     (-> matrix matrix matrix)
     (-> matrix matrix Number matrix)
     (-> matrix matrix Number Number matrix)))
(define matrix-matmul
  (case-lambda
    [([m1 : matrix] [m2 : matrix])
     (matrix-matmul m1 m2 (ann 1 Number) (ann 0 Number))]
    [([m1 : matrix] [m2 : matrix] [a : Number])
     (matrix-matmul m1 m2 a (ann 0 Number))]
    [([m1 : matrix] [m2 : matrix] [a : Number] [b : Number])
       (define r1 (matrix-rows m1))
       (define c1 (matrix-cols m1))
       (define r2 (matrix-rows m2))
       (define c2 (matrix-cols m2))
       (unless (= c1 r2)
         (error "matrix dims must match for matmul"))
       ;; recursive dot product of row i from m1 and column j from m2
       (: dot-product (Integer Integer Integer -> Number))
       (define (dot-product i j k)
         (if (= k 0)
             0
             (+ (* (vector-ref (matrix-data m1) (+ (* i c1) (- k 1)))
                   (vector-ref (matrix-data m2) (+ (* (- k 1) c2) j)))
                (dot-product i j (- k 1)))))

       ;; Build the result matrix data vector
       (define result-data
         (build-vector (* r1 c2)
                       (λ: ([i : Integer]) : Number
                         (let ([row (quotient i c2)]
                               [col (remainder i c2)])
                           (+ (* a (dot-product row col c1)) b)))))

       (make-matrix c2 r1 result-data)]))



(define A (make-matrix 3 2 #(1 2 3 4 5 6))) ; 2×3 matrix
(define B (make-matrix 2 3 #(7 8 9 10 11 12))) ; 3×2 matrix

(define C (matrix-matmul A B)) ; Expected: 2×2 matrix
(check-equal? (matrix-cols C) 2)
(check-equal? (matrix-rows C) 2)
(check-equal? (matrix-data C) #(58 64 139 154))

(define I (make-matrix 2 2 #(1 0 0 1)))
(define M (make-matrix 2 2 #(5 6 7 8)))

(define R (matrix-matmul I M))
(check-equal? (matrix-data R) #(5 6 7 8))

(define M2 (make-matrix 2 2 #(1 2 3 4)))
(define R2 (matrix-matmul M2 I 2))
(check-equal? (matrix-data R2) #(2 4 6 8))



