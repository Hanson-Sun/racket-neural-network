#lang typed/racket
(require rackunit)
(require/typed rackunit
  [check-equal? (-> Any Any Void)]
  [check-= (-> Real Real Real Void)])
(require "matrix.rkt")

(provide ActivationFn)
(define-type ActivationFn (-> matrix (Listof Real) matrix))

;; activations
(provide sigmoid)
(: sigmoid ActivationFn)
(define (sigmoid m opt)
  (define cols (matrix-cols m))
  (define rows (matrix-rows m))
  (define v (matrix-data m))
  (make-matrix cols rows (build-vector (* cols rows) (λ: ([i : Integer])
                                                       (define x (vector-ref v i))
                                                       (/ 1 (+ 1 (exp (- x))))))))

(provide sigmoid-p)
(: sigmoid-p ActivationFn)
(define (sigmoid-p m opt)
  (define cols (matrix-cols m))
  (define rows (matrix-rows m))
  (define v : (Vectorof Real) (matrix-data m))
  (define result
    (build-vector (* cols rows)
      (λ: ([i : Integer]) : Real
        (define x (vector-ref v i))
        (define s (/ 1 (+ 1 (exp (- x)))))
        (* s (- 1 s)))))
  (make-matrix cols rows result))

(define m1 (make-matrix 1 1 #(0)))
(define s1 (sigmoid m1 '()))
(check-equal? (matrix-cols s1) 1)
(check-equal? (matrix-rows s1) 1)
(check-= (vector-ref (matrix-data s1) 0) 0.5 1e-6)
(define m2 (make-matrix 2 2 #(-1 0 1 2)))
(define s2 (sigmoid m2 '()))
(define expected #(0.268941 0.5 0.731059 0.880797))
(for ([i (in-range 4)])
  (check-= (vector-ref (matrix-data s2) i)
           (vector-ref expected i)
           1e-5))


(provide relu)
(: relu ActivationFn)
(define (relu [m : matrix] [opt : (Listof Real) '()])
  (define a (if (null? opt) 1.0 (first opt)))
  (define cols (matrix-cols m))
  (define rows (matrix-rows m))
  (define v (matrix-data m))
  (define result-data
    (build-vector (* cols rows)
      (λ: ([i : Integer]) : Real
        (define x (vector-ref v i))
        (max 0 (* a x)))))
  (make-matrix cols rows result-data))

(provide relu-p)
(: relu-p ActivationFn)
(define (relu-p m opt)
  (define a (if (null? opt) 1.0 (first opt)))
  (define cols (matrix-cols m))
  (define rows (matrix-rows m))
  (define v (matrix-data m))
  (define result-data
    (build-vector (* cols rows)
      (λ: ([i : Integer]) : Real
        (define x (vector-ref v i))
        (if (>= x 0) a 0))))
  (make-matrix cols rows result-data))

(provide leaky-relu)
(: leaky-relu ActivationFn)
(define (leaky-relu [m : matrix] [opt : (Listof Real) '()])
  (define a (if (null? opt) 1.0 (first opt)))       
  (define b (if (< (length opt) 2) 0.01 (second opt))) 
  (define cols (matrix-cols m))
  (define rows (matrix-rows m))
  (define v (matrix-data m))
  (define result-data
    (build-vector (* cols rows)
      (λ: ([i : Integer]) : Real
        (define x (vector-ref v i))
        (if (>= x 0)
            (* a x)
            (* b x)))))
  (make-matrix cols rows result-data))

(provide leaky-relu-p)
(: leaky-relu-p ActivationFn)
(define (leaky-relu-p [m : matrix] [opt : (Listof Real) '()])
  (define a (if (null? opt) 1.0 (first opt)))       
  (define b (if (< (length opt) 2) 0.01 (second opt))) 
  (define cols (matrix-cols m))
  (define rows (matrix-rows m))
  (define v (matrix-data m))
  (define result-data
    (build-vector (* cols rows)
      (λ: ([i : Integer]) : Real
        (define x (vector-ref v i))
        (if (>= x 0)
            a
            b))))
  (make-matrix cols rows result-data))

;; peculiar softmax function
(provide softmax)
(: softmax (-> matrix matrix))
(define (softmax m)
  (define n : Integer (matrix-rows m))
  (define cols : Integer (matrix-cols m))
  (unless (= cols 1)
    (error "softmax expects a single-column matrix (vector)"))

  (define input : (Vectorof Real) (matrix-data m))

  ;; softmax max trick for numerical stability
  (define max-val
    (for/fold ([max-val : Real (vector-ref input 0)])
              ([i : Integer (in-range n)])
      (max max-val (vector-ref input i))))

  ;; compute exp(x_i - max)
  (define exp-v
    (build-vector n
                  (λ: ([i : Integer]) : Real
                    (exp (- (vector-ref input i) max-val)))))

  ;; sum of exponentials
  (define sum-exp
    (for/fold ([s : Real 0.0])
              ([i : Integer (in-range n)])
      (+ s (vector-ref exp-v i))))

  ;; exp(x_i - max) / sum_exp
  (define output
    (build-vector n
                  (λ: ([i : Integer]) : Real
                    (/ (vector-ref exp-v i) sum-exp))))

  (make-matrix 1 n output))


;; yeah this is gonna suck
(provide softmax-p)
(: softmax-p (-> matrix matrix))
(define (softmax-p m)
  (define n : Integer (matrix-rows m))
  (define cols : Integer (matrix-cols m))
  (unless (= cols 1)
    (error "softmax-p expects a single-column matrix (vector)"))

  (define s : (Vectorof Real) (matrix-data m))

  ;; n x n jacobian
  (define jacobian-data
    (build-vector (* n n)
                  (λ: ([i : Integer]) : Real
                    (define row (quotient i n))
                    (define col (remainder i n))
                    (define si (vector-ref s row))
                    (define sj (vector-ref s col))
                    (* si (if (= row col) (- 1 sj) (- 0 sj))))))

  (make-matrix n n jacobian-data))

(provide activation-derivatives)
(: activation-derivatives (HashTable ActivationFn ActivationFn))
(define activation-derivatives
  (hash
    sigmoid sigmoid-p
    relu relu-p
    leaky-relu leaky-relu-p
    softmax softmax-p))