import unittest
import numpy as np
import torch
from model import Reinker,compose
from prepare import features,old_bounds


class PilotTests(unittest.TestCase):
    def test_blank_and_border(self):
        x=np.zeros((33,41),np.float32)
        f,s=features(x)
        self.assertEqual(float(s.max()),0)
        raw,p=compose(torch.tensor(f[None]),torch.ones(1,1,33,41),torch.tensor(s[None,None]))
        self.assertEqual(float(p.max()),0)
        x[16,20]=1
        f,s=features(x)
        _,p=compose(torch.tensor(f[None]),torch.ones(1,1,33,41),torch.tensor(s[None,None]))
        self.assertEqual(float(p[...,:8,:].max()),0)
        self.assertEqual(float(p[...,0,0]),0)

    def test_native_shape_and_identity_initialization(self):
        m=Reinker(8)
        f=torch.rand(1,3,33,41)
        r=m(f)
        self.assertEqual(tuple(r.shape),(1,1,33,41))
        self.assertEqual(float(r.detach().abs().max()),0)
        _,p=compose(f,r,torch.ones(1,1,33,41))
        self.assertTrue(torch.equal(p,f[:,:1]))

    def test_both_gradient_directions_are_alive(self):
        for direction in (-1,1):
            m=Reinker(8)
            f=torch.full((1,3,24,24),.5)
            raw,_=compose(f,m(f),torch.ones(1,1,24,24))
            y=torch.full_like(raw,.5+.2*direction)
            (raw-y).square().mean().backward()
            self.assertGreater(float(m.head.weight.grad.abs().sum()),0)
            self.assertLess(float(m.head.bias.grad)*direction,0)

    def test_relaxed_band_can_add_and_remove(self):
        x=np.zeros((65,65),np.float32)
        x[32,20:45]=1
        f,s=features(x)
        f=torch.tensor(f[None]);s=torch.tensor(s[None,None])
        _,p=compose(f,torch.ones(1,1,65,65),s)
        self.assertEqual(float(p[0,0,40,32]),1)
        _,p=compose(f,-torch.ones(1,1,65,65),s)
        self.assertEqual(float(p[0,0,32,32]),0)  # Capability, NOT safety guarantee.


if __name__=='__main__':
    torch.set_num_threads(2)
    unittest.main()
