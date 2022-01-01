#python scripts/decomposer.py --model_config decomposer/test/se_decomposer_mag.yaml
#python scripts/decomposer.py --model_config decomposer/test/se_decomposer_magpha.yaml
#python scripts/decomposer.py --model_config decomposer/test/se_decomposer.yaml

python scripts/decomposer.py --model_config decomposer/augment/decomposer1L255_nores_bot64_n64.yaml
python scripts/decomposer.py --model_config decomposer/augment/se_exception_at_nosep.yaml
python scripts/decomposer.py --model_config decomposer/augment/se_exception_noat_nosep.yaml
python scripts/decomposer.py --model_config decomposer/augment/se_simple_at_nosep.yaml
python scripts/decomposer.py --model_config decomposer/augment/se_simple_noat_nosep.yaml
python scripts/decomposer.py --model_config decomposer/augment/se_simple_noat_sep.yaml
python scripts/decomposer.py --model_config decomposer/augment/se1024_simple_noat_nosep.yaml

python scripts/decomposer.py --model_config decomposer/augment2d/timexception_selu_noat_mag.yaml
python scripts/decomposer.py --model_config decomposer/augment2d/timexception_selu_noat_magpha.yaml
python scripts/decomposer.py --model_config decomposer/augment2d/timexception_selu_at_mag.yaml
python scripts/decomposer.py --model_config decomposer/augment2d/timexception_selu_at_magpha.yaml

python scripts/decomposer.py --model_config decomposer/base/decomposer1L.yaml
python scripts/decomposer.py --model_config decomposer/base/decomposer1L255_nores.yaml
python scripts/decomposer.py --model_config decomposer/base/decomposer1L255_nores_mish.yaml
python scripts/decomposer.py --model_config decomposer/base/decomposer1L255_nores_tanh.yaml
python scripts/decomposer.py --model_config decomposer/base/decomposer3L.yaml
python scripts/decomposer.py --model_config decomposer/base/decomposer3L255_nores.yaml
python scripts/decomposer.py --model_config decomposer/base/decomposer3L_nores.yaml


python scripts/decomposer.py --model_config karaoke/augment/decomposer1L255_nores_bot64_n64.yaml
python scripts/decomposer.py --model_config karaoke/augment/se_exception_at_nosep.yaml
python scripts/decomposer.py --model_config karaoke/augment/se_exception_noat_nosep.yaml
python scripts/decomposer.py --model_config karaoke/augment/se_simple_at_nosep.yaml
python scripts/decomposer.py --model_config karaoke/augment/se_simple_noat_nosep.yaml
python scripts/decomposer.py --model_config karaoke/augment/se_simple_noat_sep.yaml
python scripts/decomposer.py --model_config karaoke/augment/se1024_simple_noat_nosep.yaml

python scripts/decomposer.py --model_config karaoke/augment2d/timexception_selu_noat_mag.yaml
python scripts/decomposer.py --model_config karaoke/augment2d/timexception_selu_noat_magpha.yaml
python scripts/decomposer.py --model_config karaoke/augment2d/timexception_selu_at_mag.yaml
python scripts/decomposer.py --model_config karaoke/augment2d/timexception_selu_at_magpha.yaml
