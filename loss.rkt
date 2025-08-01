#lang typed/racket
(require rackunit)
(require/typed rackunit
  [check-equal? (-> Any Any Void)]
  [check-= (-> Real Real Real Void)])
(require "matrix.rkt")