
import jax.numpy as jnp
import jax
import unittest

import unittest
from frp.orthogonal import detect_identity_matrices, random_choice, create_orthogonal_matrices, create_words, create_words_ex

class TestOrthogonal(unittest.TestCase):
    def test_detect_identity_matrices(self):

        key = jax.random.PRNGKey(0)
        N, D = 5, 3
        array = jax.random.uniform(key, (N, D, D))

        # いくつかの行列を単位行列に置き換える
        array = array.at[1].set(jnp.eye(D))
        array = array.at[3].set(jnp.eye(D))

        result = detect_identity_matrices(array)
        self.assertTrue( (result == jnp.array([1,3])).all())

    def test_new_ditm(self):


        def create_words(matrices, depth, in_size=2, out_size=4, max_depth=6):
            def create_word(i):
                word = jnp.eye(out_size)
                for j in range(depth):
                    index = (i >> (j * (max_depth // depth))) & (2**(max_depth // depth) - 1)
                    word = word @ matrices[index]
                return word
            
            result = jax.vmap(create_word)(jnp.arange(2**max_depth))
            exclude = detect_identity_matrices(result)
            
            return result[:, :in_size, :], exclude

        # 使用例
        key = jax.random.PRNGKey(0)
        matrices = jax.random.uniform(key, (16, 4, 4))  # 16個の4x4行列
        depth = 3
        in_size = 2
        out_size = 4
        max_depth = 6

        words, excluded = create_words(matrices, depth, in_size, out_size, max_depth)
        print("Words shape:", words.shape)
        print("Excluded indices:", excluded)



class TestRandomChoice(unittest.TestCase):
    def setUp(self):
        self.key = jax.random.PRNGKey(0)

    def test_basic_functionality(self):
        total_words = 10
        exclude = jnp.array([2, 5, 8])
        result = random_choice(self.key, total_words, exclude)
        self.assertNotIn(result, exclude)
        self.assertTrue(0 <= result < total_words)

    def test_all_excluded(self):
        total_words = 5
        exclude = jnp.arange(5)
        result = random_choice(self.key, total_words, exclude)
        self.assertEqual(result, -1)


    def test_distribution(self):
        total_words = 10
        exclude = jnp.array([2, 5, 8])
        n_samples = 1000
        results = jnp.array([random_choice(jax.random.fold_in(self.key, i), total_words, exclude) for i in range(n_samples)])
        unique, counts = jnp.unique(results, return_counts=True)
        self.assertEqual(len(unique), total_words - len(exclude))
        self.assertTrue(jnp.all(counts > 0))  # All non-excluded options should be chosen at least once



class TestOrthogonalOperations(unittest.TestCase):
    def setUp(self):
        self.key = jax.random.PRNGKey(0)
        self.size = 4
        self.depth = 2
        self.max_depth = 8

    def test_create_orthogonal_matrices_with_adjoint(self):
        matrices = create_orthogonal_matrices(self.key, self.depth, size=self.size, max_depth=self.max_depth, with_adjoint=True)
        self.assertEqual(matrices.shape, (2**(self.max_depth // self.depth), self.size, self.size))
        
        # Check orthogonality
        for i in range(matrices.shape[0]):
            self.assertTrue(jnp.allclose(jnp.dot(matrices[i], matrices[i].T), jnp.eye(self.size), atol=1e-6))
        
        # Check that adjoints are included
        for i in range(0, matrices.shape[0], 2):
            self.assertTrue(jnp.allclose(matrices[i+1], matrices[i].T, atol=1e-6))

    def test_create_words_with_identity(self):
        matrices = create_orthogonal_matrices(self.key, self.depth, size=self.size, max_depth=self.max_depth, with_adjoint=True)
        words, exclude = create_words_ex(matrices, self.depth, in_size=2, out_size=self.size, max_depth=self.max_depth)
        
        self.assertEqual(words.shape, (2**self.max_depth, self.size, self.size))
        
        # Check that identity matrix is included in words
        identity_found = False
        for i in range(words.shape[0]):
            if jnp.allclose(words[i], jnp.eye(self.size), atol=1e-6):
                identity_found = True
                break
        self.assertTrue(identity_found)
        
        # Check that exclude contains the index of the identity matrix
        self.assertTrue(len(exclude) > 0)
        self.assertTrue(jnp.any(jnp.allclose(words[exclude], jnp.eye(self.size), atol=1e-6)))

    def test_random_choice_excludes_identity(self):
        matrices = create_orthogonal_matrices(self.key, self.depth, size=self.size, max_depth=self.max_depth, with_adjoint=True)
        words, exclude = create_words_ex(matrices, self.depth, in_size=2, out_size=self.size, max_depth=self.max_depth)
        
        n_samples = 1000
        results = jnp.array([random_choice(jax.random.fold_in(self.key, i), words.shape[0], exclude) for i in range(n_samples)])
        
        # Check that no choice is in the excluded set
        self.assertTrue(jnp.all(jnp.isin(results, exclude, invert=True)))
        
        # Check that identity matrix is never chosen
        for result in results:
            self.assertFalse(jnp.allclose(words[result], jnp.eye(self.size), atol=1e-6))
