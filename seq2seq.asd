#|
  This file is a part of seq2seq project.
|#

(defsystem "seq2seq"
  :depends-on (:reader
               :cl-utilities
               :py4cl2
               :iterate)
  :components ((:file "seq2seq"))
  :description "a remake of keras seq2seq tutorial
https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html")
