python -m experiments.classification.adult_predictions --definition DisparateImpact    --e -0.80 --n_iters 5000 --r_train_v_test 0.7 --r_cand_v_safe 0.4 --d 0.1 
python -m experiments.classification.adult_predictions --definition EqualizedOdds      --e 0.1  --n_iters 5000 --r_train_v_test 0.7 --r_cand_v_safe 0.4 --d 0.05 
python -m experiments.classification.adult_predictions --definition EqualOpportunity   --e 0.1   --n_iters 5000 --r_train_v_test 0.4 --r_cand_v_safe 0.4 --d 0.05 