from collections import defaultdict
from itertools import product
import numpy as np
from tqdm import tqdm

def count_keys(bitting_positions: int, bitting_depths: int, macs: int) -> int:
  print(f'Physical keys with n={bitting_positions}, k={bitting_depths}, macs={macs}')
  if bitting_positions == 0:
    return 1
  # For convenience, we'll refer to bitting_positions as n.
  #
  # Let's define a proposition about the natural numbers:
  #
  #  P(k): For each i, j, A^k_{i, j} contains the number of ways
  #  to create keys of length k + 1, which start at bitting depth
  #  i, and end at bitting depth j.
  #
  # We'll prove P is true for all natural numbers, by induction.
  #
  # To see P(1), note that A^1 = A, and so A_{i, j} should be 1 if
  # a key of length 2, with depths i, j, is MACS-valid. That is,
  # if |i - j| <= macs. This is precisely how we construct A:
  A = np.fromfunction(lambda i, j: (abs(i - j) <= macs).astype('object'),
                      (bitting_depths, bitting_depths))
  # The .astype('object') is because we want to use Python's
  # arbitrary-length integers, and not something like `int32`,
  # which will overflow for large numbers of combinations.
  #
  # Assuming P(k), let's show P(k + 1). Note A^{k + 1} = A^k @ A,
  # where @ means matrix multiplication. By definition of matrix
  # multiplication, this means:
  #   A^{k + 1}_{i, j} = \sum_{l = 1}^n A^k_{i, l} * A_{l, j}
  # By P(k), we know A^k_{i, l} is the number of ways to create keys
  # of length k + 1, that start at bitting depth i, and end at bitting
  # depth l. If we then multiply that by A_{l, j}, which is 1 iff
  # |l - j| <= macs, we get the number of ways to create keys of
  # length k + 2, that start at bitting depth i, and have l, j as
  # their last two bitting depths. If we then sum over all l
  # between 1 and n, giving A^{k + 1}, these are all the ways
  # of constructing keys of length k + 2, which start at bitting
  # depth i, and end at bitting depth j. This is the statement
  # of P(k + 1), which was to be shown.
  #
  # Thus, to compute the number of keys of length n, we need only
  # compute A^{n - 1}, and sum over all its i, j coordinates, since
  # we're interested in keys of length n that start with any
  # bitting depth i, and end with any bitting depth j.
  return np.linalg.matrix_power(A, bitting_positions - 1).sum()

def brute(n, d, macs):
  
  z = product(range(d), repeat=n)
  def good(t):
    freqs = defaultdict(int)
    for i in range(n):
      freqs[t[i]] += 1
    for f in freqs.values():
      if f > n/2: return False
    for i in range(n - 1):
      if abs(t[i] - t[i + 1]) > macs: return False
    for i in range(n - 2):
      if t[i] == t[i + 1] == t[i + 2]: return False
    return True
  
  return sum(1 for t in z if good(t))

def count_keys_macs_and_en1303(n, d, macs):
  print(f'Legal keys ith {n=}, {d=}, {macs=}')
  # The algorithm is a breadth first search on the state space of keys.
  # The state space is defined as the set of pairs (f, (a, b)), where f
  # is a table of frequencies of depths, and (a, b) are the last two
  # bitting depths in a key. State transitions are adding a bitting at
  # the end of a key, which changes the frequency table as well as the
  # last two depths.
  # 
  # We start with a key of length zero, which has a frequency of 0 for
  # all depths, and None as its last two depths.
  initial_state = ((0,) * d, (None, None))
  # We will then remember how many ways there are to make a key in each
  # state. For the initial state, the only way to reach it is via a key
  # of length zero, so paths[0] = 1.
  id_to_state = {0: initial_state}
  paths = defaultdict(int, {0: 1})

  # This helper tells us, given a state, what are all the possible states
  # one can reach from it.
  def neighbors(state):
    (freqs, (d1, d2)) = state
    if sum(freqs) == n: return
    # For each possible bitting depth:
    for b in range(d):
      # If adding this cut at the end of a key like this would create three
      # identical cuts, it is not a valid transition.
      if (d1 == d2 == b): continue
      # If adding this cut at the end of a key like this would violate MACS,
      # it is not a valid transition.
      if d2 is not None and abs(d2 - b) > macs: continue
      # If adding this cut at the end of a key like this would mean one depth
      # accounts for more than half of the entire key's depths, it is not a
      # valid transition. Note we do not compare against the _current_, shorter
      # key length here, but against the length of the entire, eventual key,
      # which is n.
      if freqs[b] + 1 > n / 2: continue
      # The next frequency table is the same as this one, except we add a cut
      # of depth b, so we must increase the frequency of b by one.
      next_freqs = (*freqs[:b], freqs[b] + 1, *freqs[b + 1:])
      # If the key we started from ended in (d1, d2), and we add the cut b at
      # the end, it now ends in (d2, b).
      next_state = (next_freqs, (d2, b))
      yield next_state
  
  # We start from the initial state, representing all (one) keys of length
  # zero. The queue will contain the states a key of length i can be in,
  # for each i in the loop below.
  q = [0]
  pbar1 = tqdm(total=n, desc="Loops", position=0)
  # We iterate the following procedure n times, which adds a cut at the end
  # of all known key states.
  for i in range(n):
    # The following variables are meaningful only for the next level in the
    # graph. ne_q, for example, means the queue of nodes in the next level
    # of the breadth first search in the state space graph. These will be
    # the states a key of length i + 1 can be in.
    new_q = []
    new_state_to_id = {}
    new_id_to_state = {}
    new_paths = defaultdict(int)
    pbar2 = tqdm(total=len(q), desc="Search", position=1, leave=False)
    # Main breadth first search loop.
    while q:
      u_id = q.pop()
      u = id_to_state[u_id]
      # We visit every neighbor of u, that is, a state v that can be reached
      # from state u by adding a single cut at the end of any key in state u.
      for v in neighbors(u):
        if v not in new_state_to_id:
          # If we've not seen this state before, this is a new node in the
          # graph, at level i + 1 in the breadth first search. Thus, we add
          # it to the new_q, to loop over in the next iteration.
          v_id = len(new_state_to_id)
          new_state_to_id[v] = v_id
          new_id_to_state[v_id] = v
          new_q.append(v_id)
        else:
          v_id = new_state_to_id[v]
        # Every way to reach the state u, gives one more way to reach the
        # state v, by adding a single cut at the end of those keys.
        new_paths[v_id] += paths[u_id]
      pbar2.update()
    q = new_q
    id_to_state = new_id_to_state
    paths = new_paths
    pbar1.update()
    pbar1.refresh()
  pbar2.close()
  print("")
  # All the states that are now in paths are states of keys of length n.
  # Thus, we can sum over all of the ways of reaching all of these states,
  # and since each such way is one distinct key, we'll get all of the possible
  # keys of length n that meet our requirements.
  return sum(paths.values())

print(count_keys(14, 8, 6))
print(count_keys_macs_and_en1303(14, 8, 6))
