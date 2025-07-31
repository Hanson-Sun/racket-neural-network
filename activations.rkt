#lang typed/racket
(require rackunit)
(require/typed rackunit
  [check-equal? (-> Any Any Void)]
  [check-= (-> Number Number Number Void)])
(require "matrix.rkt")


;; activations
(provide sigmoid)
(: sigmoid (matrix -> matrix))
(define (sigmoid m)
  (define cols (matrix-cols m))
  (define rows (matrix-rows m))
  (define v (matrix-data m))
  (make-matrix cols rows (build-vector (* cols rows) (λ: ([i : Integer])
                                                       (define x (vector-ref v i))
                                                       (/ 1 (+ 1 (exp (- x))))))))

(define m1 (make-matrix 1 1 #(0)))
(define s1 (sigmoid m1))
(check-equal? (matrix-cols s1) 1)
(check-equal? (matrix-rows s1) 1)
(check-= (vector-ref (matrix-data s1) 0) 0.5 1e-6)
(define m2 (make-matrix 2 2 #(-1 0 1 2)))
(define s2 (sigmoid m2))
(define expected #(0.268941 0.5 0.731059 0.880797))
(for ([i (in-range 4)])
  (check-= (vector-ref (matrix-data s2) i)
           (vector-ref expected i)
           1e-5))