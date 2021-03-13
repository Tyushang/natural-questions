#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# __author__=u"Frank Jing"

import numpy as np
import pandas as pd


aa = np.random.random(size=[8,2])*10

df = pd.DataFrame(aa, columns=['foo', 'bar', ])
df['cate'] = ['a', 'b', 'a', 'c', 'd', 'b', 'a', 'c']
df['nan'] = [1, 2, 3, np.nan, 3, 4, 5, np.nan]
df['info'] = ['chengdu', 'shanghai', 'hangzhou', 'wuhan', 'shenzhen', 'harbin', 'chongqing', 'xian']


#
# class Foo:
#
#     def __call__(self, *args, **kwargs):
#         print(args)
#         print(kwargs)

