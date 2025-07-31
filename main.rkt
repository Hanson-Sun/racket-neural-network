#lang typed/racket
(require rackunit)

(require/typed rackunit
  [check-equal? (-> Any Any Void)]
  [check-= (-> Number Number Number Void)])