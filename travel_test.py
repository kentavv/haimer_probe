#!/usr/bin/env python

import math

def min_path(lst, start_pt=None, end_pt=None):
    from itertools import permutations 

    m_d = float('inf')
    m_lst = []
    for l in permutations(lst):
      l2 = list(l)

      if start_pt is not None:
          l2 = [start_pt] + l2

      if end_pt is not None:
          l2 = l2 + [end_pt]

      d = 0.
      for pt1, pt2 in zip(l2, l2[1:]):
          d += (pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2

      if m_d > d:
         m_d = d
         m_lst = l2

      print(l2)
      print(zip(l2, l2[1:]))
      print(d)

    print(m_d, m_lst)

    return math.sqrt(m_d), m_lst


# Optionally, past the current machine position, then the position of the holes, and finally the desired end machine position.

lst = [(1, .5), (2,1), (2,2), (4,0), (.5,.5)]
print(min_path(lst, start_pt = (0,0), end_pt=(1,1)))
