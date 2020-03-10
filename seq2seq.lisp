;; Local Variables:
;; eval: (put 'defparameters 'lisp-indent-function 0)
;; eval: (put 'dlet* 'lisp-indent-function 1)
;; End:

(eval-when (:compile-toplevel :load-toplevel)
  (py4cl2:defpymodule "numpy")
  (py4cl2:defpymodule "pickle")
  (py4cl2:defpymodule "keras" t))

(defpackage :seq2seq
  (:use :cl :reader :iterate :py4cl2)
  (:shadowing-import-from :keras.layers :dense/class :lstm/class :input/1)
  (:shadowing-import-from :keras.models :load-model)
  (:shadowing-import-from :keras :model/class)
  (:shadowing-import-from :cl-utilities :split-sequence))

(in-package :seq2seq)

(eval-when (:compile-toplevel :load-toplevel :execute)
  (setf (symbol-function 'lstm) #'lstm/class
	(symbol-function 'model) #'model/class
	(symbol-function 'dense) #'dense/class))


(defmacro defparameters (&rest args)
  `(progn
     ,@(loop for arg in args
          collect `(defparameter ,@arg))))

(defmacro dlet* (var-list &body body)
  (if var-list
      (macroexpand
       (etypecase (caar var-list)
         (list `(destructuring-bind ,(caar var-list) ,(cadar var-list)
                  ,`(dlet* ,(cdr var-list) ,@body)))
         (symbol `(let (,(car var-list)) ,`(dlet* ,(cdr var-list) ,@body)))))
      `(progn ,@body)))

(defparameters
    (*batch-size* 32 "Batch size for training.")
    (*epochs* 40 "Batch size for Number of epochs to train for.")
  (*latent-dim* 512 "Latent dimensionality of the encoding space.")
  (*num-samples* 1000 "Number of samples to train on.")
  (*data-path* "DATASET.txt")
  ;; (*data-path* "fra.txt")
  (*validation-split* 0.2))

(defun define-compiled-model (num-encoder-tokens num-decoder-tokens latent-dim)

  (let* ((encoder-inputs (input/1 :shape `("None" ,num-encoder-tokens)))
         (encoder (lstm :units latent-dim
                        :return-state t))
         (encoder-lstm (keras.layers:bidirectional/class
                        :layer (lstm :units latent-dim
                                     :return-state t)))
         (encoder-output-with-states (pycall encoder encoder-inputs))
         ;; (encoder-outputs (pyeval encoder-output-with-states "[0]"))
         (encoder-states (pyeval encoder-output-with-states "[1:]")) 

         (decoder-inputs (input/1 :shape `("None" ,num-decoder-tokens)))
         (decoder-lstm (lstm :units latent-dim
                             :return-state t
                             :return-sequences t))
         (decoder-outputs-after-lstm
          (get-val (pycall decoder-lstm
                           decoder-inputs
                           :initial-state encoder-states)
                   0)) ;; discarding states here
         (decoder-dense (dense :units num-decoder-tokens
                               :activation "softmax"))
         (decoder-outputs (pycall decoder-dense
                                  decoder-outputs-after-lstm))
         (model (model (vector encoder-inputs decoder-inputs) decoder-outputs)))
    (pymethod model 'compile :optimizer "rmsprop" :loss "categorical_crossentropy")
    model))

(defun train-model (model
                    encoder-input-data decoder-input-data decoder-target-data
                    &key (batch-size *batch-size*)
                      (epochs *epochs*)
                      (validation-split *validation-split*))
  (pymethod model 'fit
            (vector encoder-input-data decoder-input-data)
            decoder-target-data
            :batch-size batch-size
            :epochs epochs
            :validation-split validation-split))

(let ((input-token-index-table (make-hash-table))    ; sets
      (target-token-index-table (make-hash-table))
      (input-token-list nil)
      (target-token-list nil)
      (max-encoder-seq-length nil)
      (max-decoder-seq-length nil))

  (defun get-texts (&key (data-path *data-path*) (max-num-samples *num-samples*))
    "Split text into the two languages, and pad the target text with #\Tab 
and #\Newline characters. Returns a list of num-samples, input-texts and
target-texts"
    (iter (for line in-file data-path using #'read-line)
          (for i below max-num-samples)
          (for (input-text target-text) = (split-sequence #\Tab line))
          (setq target-text (with-output-to-string (str)
                              (write-char #\Tab str)
                              (write-string target-text str)
                              (write-char #\Newline str)))

          (collect input-text into input-texts)
          (collect target-text into target-texts)
          (finally (format t "I: ~D~%" i) (return (list i input-texts target-texts)))))

  (defun process-texts-to-characters (input-texts target-texts)
    "Populates token-index-table and token-list."
    (iter outer
          (for input-text in input-texts)
          (generate j from 0)
          (iter (for input-char in-string input-text)
                (unless (get-val input-token-index-table input-char)
                  (setf (get-val input-token-index-table input-char)
                        (in outer (next j)))
                  (setq input-token-list (cons input-char input-token-list))))
          (finally (setq input-token-list (reverse input-token-list ))
                   (format t "INPUT-TOKEN-LIST: ~D~%" input-token-list)))
    (setq max-encoder-seq-length (apply #'max (mapcar #'length input-texts)))
    (iter outer
          (for target-text in target-texts)
          (generate j from 0)
          (iter (for target-char in-string target-text)
                (unless (get-val target-token-index-table target-char)
                  (setf (get-val target-token-index-table target-char)
                        (in outer (next j)))
                  (setq target-token-list (cons target-char target-token-list))))
          (finally (setq target-token-list (reverse target-token-list))
                   (format t "TARGET-TOKEN-LIST: ~D~%" target-token-list)))
    (setq max-decoder-seq-length (apply #'max (mapcar #'length target-texts)))
    t)

  (defun get-num-tokens ()
    (list (hash-table-count input-token-index-table)
          (hash-table-count target-token-index-table)))
  (defun get-token-lists ()
    (list input-token-list target-token-list))

  (defun get-input-char-with-index (index) (get-val input-token-list index))
  (defun get-target-char-with-index (index) (get-val target-token-list index))

  (defun get-index-of-input-char (char) (get-val input-token-index-table char))
  (defun get-index-of-target-char (char) (get-val target-token-index-table char))

  (defun get-max-token-lengths ()
    (list max-encoder-seq-length max-decoder-seq-length))

  (defun input-text-to-3d-array (input-text)
    (let ((arr (make-array (list 1
                                 max-encoder-seq-length
                                 (first (get-num-tokens)))
                           :element-type 'single-float)))
      (iter (for char in-string input-text)
            (for i upfrom 0)
            (setf (get-val arr 0 i (get-val input-token-index-table char))
                  1.0))
      arr))
  (defun target-text-to-3d-array (target-text)
    (let ((arr (make-array (list 1
                                 max-decoder-seq-length
                                 (second (get-num-tokens)))
                           :element-type 'single-float)))
      (iter (for char in-string target-text)
            (for i upfrom 0)
            (setf (get-val arr 0 i (get-val target-token-index-table char))
                  1.0))
      arr))
  
  (defun prepare-and-get-data (&key (data-path *data-path*) (max-num-samples *num-samples*))
    (destructuring-bind (num-samples input-texts target-texts)
        (get-texts :data-path data-path :max-num-samples max-num-samples)
      (process-texts-to-characters input-texts target-texts)
      (let* ((num-encoder-tokens (hash-table-count input-token-index-table))
             (num-decoder-tokens (hash-table-count target-token-index-table))
             (encoder-input-data
              (make-array (list num-samples max-encoder-seq-length num-encoder-tokens)
                          :element-type 'single-float :initial-element 0.0))
             (decoder-input-data
              (make-array (list num-samples max-decoder-seq-length num-decoder-tokens)
                          :element-type 'single-float :initial-element 0.0))
             (decoder-target-data
              (make-array (list num-samples max-decoder-seq-length num-decoder-tokens)
                          :element-type 'single-float :initial-element 0.0)))
        (format t "Number of unique input token: ~d~%" num-encoder-tokens)
        (format t "Number of unique output token: ~d~%" num-decoder-tokens)
        (format t "Max sequence length for inputs: ~d~%" max-encoder-seq-length)
        (format t "Max sequence length for outputs: ~d~%" max-decoder-seq-length)
        (iter (for input-text in input-texts)
              (for target-text in target-texts)
              (for j from 0)
              (iter (for id from 0)
                    (for input-char in-string input-text)
                    (setf (get-val encoder-input-data
                                   j id (get-val input-token-index-table
                                                 input-char))
                          1.0))
              (iter (for id from 0)
                    (for target-char in-string target-text)
                    (setf (get-val decoder-input-data
                                   j id (get-val target-token-index-table
                                                 target-char))
                          1.0)
                    (unless (first-iteration-p)
                      (setf (get-val decoder-target-data
                                     j (1- id) (get-val target-token-index-table
                                                        target-char))
                            1.0))))
        (list encoder-input-data decoder-input-data decoder-target-data)))))

(defun restore-trained-model-for-inference
    (weights-file num-encoder-tokens num-decoder-tokens latent-dim)
  "Returns a two-item list corresponding to encoder and decoder models."
  (let* ((encoder-inputs (input/1 :shape `(nil ,num-encoder-tokens)))
         (encoder (lstm :units latent-dim
                        :return-state t))
         (encoder-output-with-states (pycall encoder encoder-inputs))
         (encoder-states (pyeval encoder-output-with-states "[1:]")) 

         (decoder-inputs (input/1 :shape `(nil ,num-decoder-tokens)))
         (decoder-lstm (print (lstm :units latent-dim
                                    :return-state t
                                    :return-sequences t)))
         (decoder-outputs-after-lstm
          (get-val (pycall decoder-lstm
                           decoder-inputs
                           :initial-state encoder-states)
                   0)) ;; discarding states here
         (decoder-dense (dense :units num-decoder-tokens
                               :activation "softmax"))
         (decoder-outputs (pycall decoder-dense
                                  decoder-outputs-after-lstm))
         (model (model (vector encoder-inputs decoder-inputs) decoder-outputs)))
    (pymethod model 'load-weights weights-file)
    (pymethod model 'compile :optimizer "rmsprop" :loss "categorical_crossentropy")
    
    (list (model encoder-inputs encoder-states)
          (let* ((decoder-c (input/1 :shape `(,latent-dim)))
                 (decoder-h (input/1 :shape `(,latent-dim)))
                 (decoder-states-input (vector decoder-c decoder-h))
                 (decoder-output-with-states
                  (pycall decoder-lstm
                          decoder-inputs
                          :initial-state decoder-states-input))
                 (decoder-states (pyeval decoder-output-with-states "[1:]"))
                 (decoder-outputs (pycall decoder-dense
                                          (pyeval decoder-output-with-states "[0]"))))
            (model (vector decoder-inputs decoder-h decoder-c)
                   (pyeval "[" decoder-outputs "] +" decoder-states))))))


(defun define-encoder-model (trained-model num-encoder-tokens)
  (let* ((encoder-inputs (input/1 :shape `("None" ,num-encoder-tokens)))
         (encoder (get-val (pyslot-value trained-model 'layers) 2))
         (encoder-output-with-states (pycall encoder encoder-inputs))
         (encoder-states (pyeval encoder-output-with-states "[1:]")))
    (model encoder-inputs encoder-states)))

(defun define-decoder-model (trained-model num-decoder-tokens latent-dim)
  (let* ((decoder-inputs (input/1 :shape `("None" ,num-decoder-tokens)))
         (decoder-h (input/1 :shape `(,latent-dim)))
         (decoder-c (input/1 :shape `(,latent-dim)))
         (decoder-states-input (vector decoder-h decoder-c))
         (decoder-lstm (get-val (pyslot-value trained-model 'layers) 3))
         (decoder-output-with-states
          (pycall decoder-lstm
                  decoder-inputs
                  :initial-state decoder-states-input))
         (decoder-states (pyeval decoder-output-with-states "[1:]"))
         (decoder-dense (get-val (pyslot-value trained-model 'layers) 4))
         (decoder-outputs (pycall decoder-dense
                                  (pyeval decoder-output-with-states "[0]"))))
    (model (pyeval "[" decoder-inputs "] + " decoder-states-input)
           (pyeval "[" decoder-outputs "] +" decoder-states))))

(defun decode-sequence (input-sequence encoder-model decoder-model
                        num-decoder-tokens max-decoder-seq-length)
  
  (flet ((set-to-1 (1-index)
           (let ((arr (numpy:zeros `(1 1 ,num-decoder-tokens))))
             (setf (get-val arr 0 0 1-index) 1)
             arr)))
    
    (iter (for states-value
               initially (pymethod encoder-model 'predict input-sequence)
               then (vector h c))
          (for target-seq
               initially (set-to-1 (get-index-of-target-char #\Tab))
               then (set-to-1 sampled-char-index))
          (for decoded-sentence
               initially ""
               then (format nil "~D~D" decoded-sentence sampled-char))
          
          
          
          (for (out h c) = (coerce (pymethod decoder-model 'predict
                                             (pyeval "[" target-seq "] + " states-value))
                                   'list))
          (for sampled-char-index = (numpy:argmax :a (pyeval out "[0,-1,:]")))
          (for sampled-char = (get-target-char-with-index sampled-char-index))

          ;; (format t "INDEX: ~2D CHAR: ~D~%" sampled-char-index sampled-char)
          
          (when (or (char= sampled-char #\newline)
                    (> (length decoded-sentence) max-decoder-seq-length))
            (return decoded-sentence)))))

(defpyfun "open" "" :lisp-fun-name "PYOPEN")
(defun pickle-model-and-save (model filename)
  (let ((file (pyopen :file filename :mode "wb")))
    (pickle:dump :obj model :file file)
    (pymethod file 'close)))

(defun load-pickled-model (filename)
  (let* ((file (pyopen :file filename :mode "rb"))
         (model (pickle:load :file file)))
    (pymethod file 'close)
    model))
