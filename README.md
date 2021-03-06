# An implementation of Keras Seq2Seq Tutorial Code in Common Lisp

Python (Keras) Source: [https://github.com/keras-team/keras/blob/master/examples/lstm_seq2seq.py](https://github.com/keras-team/keras/blob/master/examples/lstm_seq2seq.py)

Let's just say, I didn't like that code - it is very imperative (than functional). How does one "reuse" it? Plus, I wanted to dabble in deep learning. 

<img src="seq2seq-lisp.png" width="80%" margin="auto"></img>

## Common Lisp Prerequisites 

Nothing hard. But these would help. Familiarity with
- [py4cl2](https://digikar99.github.io/py4cl2/) - available with [ultralisp](https://ultralisp.org/)
- [iterate](https://digikar99.github.io/cl-iterate-docs/) - 
- [cl-utilities](https://digikar99.github.io/cl-iterate-docs/) - 
- [reader](https://github.com/digikar99/reader) - ultralisp

## Warming Up

Clone this repo to somewhere `quicklisp` can find, eg. `~/quicklisp/local-projects/`.

```lisp
CL-USER> (ql:quickload "seq2seq") ; load into SLIME
CL-USER> (in-package :seq2seq)
;; Alternatively, you can `M-x slime-cd` and (push #P"./" ql:*local-project-directories*)
```

(Now, if you are using `py4cl2` for the first time, better call `(py4cl2:initialize)` for efficiency in numpy-array-transfers.) 

Once loaded, tweak the `*data-path*` to point to a "parallel-corpus" - each line of a parallel corpus contains the following in that order:

- a sentence in the language to translate from
- a tab character
- the sentence in the language to translate to

Optionally, play around more parameters: `*batch-size*, *epochs*, *latent-dim*, *num-samples*, *validation-split*`.

Once ready, as a utility, define into SLIME the following:

```lisp
SEQ2SEQ> (defun preprocess-and-compile-model ()
           (dsetq (encoder-input-data decoder-input-data decoder-target-data) (prepare-and-get-data))
           (dsetq (num-encoder-tokens num-decoder-tokens) (get-num-tokens))
           (dsetq (max-encoder-seq-length max-decoder-seq-length) (get-max-token-lengths))
           (setq model (define-compiled-model num-encoder-tokens num-decoder-tokens *latent-dim*)))
;; dsetq is a function from the iterate library - and is a combination of setq and destructuring-bind(!)

SEQ2SEQ> (defun translate (input-text)
           (decode-sequence (input-text-to-3d-array input-text)
                            encoder-model decoder-model
                            num-decoder-tokens max-decoder-seq-length))
                            
SEQ2SEQ> (defun prepare-for-translation ()
           (setq encoder-model (define-encoder-model model num-encoder-tokens))
           (setq decoder-model (define-decoder-model model num-decoder-tokens *latent-dim*)))                
```
  
I am a bit reluctant to add these to the seq2seq.lisp - these aren't "pure" functions.

## Train the model:

```lisp
SEQ2SEQ> (train-model model encoder-input-data decoder-input-data decoder-target-data)
;; this will take a while depending on the parameters - you should be able to see the results
```

## Save the model:

Now, the teacher forcing (initial-state in particular; see `(defun define-compiled-model...)` that is used here doesn't let the model be saved as expected; therefore, `keras.save_model` doesn't save all the weights. 

One workaround (that is used here) is to `pickle` the model - I know keras doesn't recommend this; but I don't know of a better method. (I did try with `save_weights` and `load_weights`; but that didn't help.)

```lisp
SEQ2SEQ> (pickle-model-and-save model "s2s.pickle")
NIL
```

## Load the model:

```lisp
SEQ2SEQ> (preprocess-and-compile-model)
;;; In case you load the wrong *data-path*, recompile the
;;; let data block (starting at line 85), and then redo (preprocess-and-compile-model).
SEQ2SEQ> (setq model (load-pickled-model "s2s.pickle"))
#S(PY4CL::PYTHON-OBJECT
   :TYPE "<class 'keras.engine.training.Model'>"
   :HANDLE 1559)
SEQ2SEQ> (prepare-for-translation)
```

## Translate something:

```lisp
SEQ2SEQ> (translate "Run!")
"Coursz   "
SEQ2SEQ> (translate "Go.")
"Va  "
```

Amuse-toi bien ! (French to english and other datasets can be found at [here](http://www.manythings.org/anki/) - though, you might have to clean it a bit.)
