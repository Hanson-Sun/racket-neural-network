#lang typed/racket
(require rackunit)
(require/typed rackunit
  [check-equal? (-> Any Any Void)]
  [check-= (-> Real Real Real Void)])

(require "matrix.rkt")
(require "activations.rkt")

(provide racket-network
         racket-network-layers
         racket-network-activations
         racket-network-weights
         racket-network-biases)
(struct racket-network ([layers : (Vectorof Integer)]
			[activations : (Vectorof ActivationFn)]
                        [weights : (Vectorof matrix)]
                        [biases : (Vectorof matrix)])
  #:transparent)

(provide make-racket-network)
(define (make-racket-network [layers : (Vectorof Integer)]
			     [activations : (Vectorof ActivationFn)]
                             [min : Real -1]
                             [max : Real 1])
  (define len (vector-length layers))
  (unless (= len (vector-length activations)) 
     (error "Real of activations must match the Real of layers"))

  (define weights (build-vector (- len 1) (λ: ([i : Integer])
                                      (make-matrix-random (vector-ref layers i)
                                                          (vector-ref layers (+ i 1))
                                                          min max))))
  
  (define biases (build-vector len (λ: ([i : Integer])
                                      (make-matrix-random 1
                                                          (vector-ref layers i)
                                                          min max))))
  (racket-network layers activations weights biases))


(provide racket-network-forward)
(: racket-network-forward (racket-network matrix -> matrix))
(define (racket-network-forward [nn : racket-network]
                                [input : matrix])
  (define weights (racket-network-weights nn))
  (define biases (racket-network-biases nn))
  (define activations (racket-network-activations nn))
  (define num-layers (vector-length weights))

  (define (forward-layer [i : Integer] [m : matrix]) : matrix
    (if (= i num-layers)
        m
        (let* ([w (vector-ref weights i)]
               [b (vector-ref biases i)]
               [act (vector-ref activations i)]
               [z (matrix-elem (matrix-matmul w m) b +)]
               [a-next (act z)])
          (forward-layer (+ i 1) a-next))))

  (forward-layer 0 input))

(provide racket-network-backward)
(: racket-network-backward (racket-network matrix -> Void))
(define (racket-network-backward [nn : racket-network]
                                        [error : matrix])
  ;; this is gonna be so grim
  (void))


(provide train)
(define (train [nn : racket-network]
               [inputs : (Vectorof matrix)]
               [outputs : (Vectorof matrix)]
               [epochs : Integer]
               [batch-size : Integer])
  (void))