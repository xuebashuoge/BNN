import unittest

import torch

from journal.diagonal_fading_stress import LAWS, law_config, paired_scores
from journal.run_experiment import evaluate, make_models, normalize_config


class PairedDeploymentTests(unittest.TestCase):
    def test_cached_encoder_matches_full_frozen_forward_and_fresh_streams(self):
        torch.set_num_threads(1)
        for name,width,dim,classes,shape in (("moons",16,2,2,(2,)),("mnist",128,784,10,(1,28,28)),
                                             ("cifar10",128,3072,10,(3,32,32))):
            cfg=normalize_config(dict(dataset=name, width=width, method="erm", predictor="gaussian",
                                architecture="cnn_spatial3" if name=="cifar10" else "mlp",
                                train_channel=law_config(.35,1),deployment_channel=law_config(.35,1),m=5))
            model,bank=make_models(cfg,dim,classes)
            rng=torch.Generator().manual_seed(11831)
            x=torch.randn(13,*shape,generator=rng)
            y=torch.randint(classes,(13,),generator=rng)
            batches=list(zip(x.split(5),y.split(5)))
            paired=paired_scores(model,bank,batches,92876,7)
            for i,(rho,std) in enumerate(LAWS):
                result=evaluate(model,bank,batches,dict(cfg,deployment_channel=law_config(rho,std)),7,92876)
                self.assertEqual(paired[i],[r["accuracy"] for r in result["replicate_statistics"]])
            first=bank.draw(300,law_config(.35,1),rng.manual_seed(9101))
            second=bank.draw(300,law_config(.35,2),rng.manual_seed(9101))
            high=bank.draw(300,law_config(.85,2),rng.manual_seed(9101))
            torch.testing.assert_close(first[0],second[0],rtol=0,atol=0)
            torch.testing.assert_close(first[1]*2,second[1],rtol=0,atol=0)
            torch.testing.assert_close(high[1],second[1],rtol=0,atol=0)
            self.assertFalse(torch.equal(high[0],second[0]))


if __name__=="__main__":
    unittest.main()
