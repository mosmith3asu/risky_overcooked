from src.risky_overcooked_rl.utils.model_manager import get_default_config, parse_args #get_argparser
from src.risky_overcooked_rl.utils.trainer import Trainer
from src.risky_overcooked_rl.utils.cirriculum import CirriculumTrainer
from src.risky_overcooked_rl.utils.deep_models import SelfPlay_QRE_OSA_CPT


if __name__ == "__main__":
    config = get_default_config()
    config = parse_args(config)
    config['p_slip'] = 0.25
    config['loads'] = 'rational'
    config['epsilon_sched'] =  [0.01, 0.01, 2000]
    config['lr_sched'] = [1e-14, 1e-14, 1_000]
    trainer = Trainer(SelfPlay_QRE_OSA_CPT, config)
    # trainer.init_sched(config)
    trainer.N_tests = 4
    trainer.test_interval = 1

    trainer.run()
    # CirriculumTrainer(SelfPlay_QRE_OSA_CPT, config).run()
    # test_reward, test_shaped_reward, state_history, action_history, aprob_history =\
    #     CirriculumTrainer(SelfPlay_QRE_OSA_CPT, config).test_rollout(rationality=10)
    # print(test_reward, test_shaped_reward)