

## Install requirements

```bash
pip install -r requirements.txt -e .
```

## Train a model

For training a plain neural network DQN for a probabilistic map with MTR method, run
```
python -m marl_model train --model dqn --env mtr --map prob --shape 10x10
```

To re-train this model with a map updates, run
```
python -m marl_model train --resume model --map prob_update --model_path <model_path>
```

For training a CNN DQN for a static map with MTNS method, run
```
python -m marl_model train --model cdqn --env mtns --map simple --shape 10x10
```

To re-train this model with a map updates, run
```
python -m marl_model train --resume model --map simple_update --model_path <model_path>
```

For more details, see
```
python -m marl_model train -h
```

## Replay a model

To replay a model, run

```
python -m marl_model replay --model_path <model_path>
```