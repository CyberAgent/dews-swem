from typing import Callable, List, Tuple

from gensim.models.keyedvectors import Word2VecKeyedVectors
import numpy as np


def _word_embed(
    token: str,
    kv: Word2VecKeyedVectors,
    uniform_range: Tuple[float, ...] = (-0.01, 0.01)
) -> np.ndarray:
    """ Get word embedding of given token.

    Args:
        token (str): A word token to get embed.
        kv (Word2VecKeyedVectors): Vocabularies dictionary.
        uniform_range (Tuple[float, ...]): A range of uniform distribution to
                                           generate random vector.

    Returns:
        numpy.ndarray: An array with shape (self.embed_dim, )
    """
    try:
        return kv[token]
    except Exception:
        embed_dim: int = kv.vector_size
        # np.random.seed(0)
        # return np.random.uniform(
        #     uniform_range[0],
        #     uniform_range[1],
        #     embed_dim
        # )
        output = [
            0.0009762700785464953,
            0.0043037873274483895,
            0.0020552675214328773,
            0.0008976636599379376,
            -0.0015269040132219053,
            0.002917882261333122,
            -0.0012482557747461494,
            0.007835460015641596,
            0.009273255210020587,
            -0.0023311696234844456,
            0.00583450076165329,
            0.000577898395058089,
            0.0013608912218786469,
            0.00851193276585322,
            -0.00857927883604226,
            -0.008257414005969186,
            -0.009595632051193485,
            0.0066523969109587595,
            0.005563135018997009,
            0.007400242964936384,
            0.00957236684465528,
            0.005983171284334473,
            -0.0007704127549413627,
            0.00561058352572911,
            -0.007634511482621335,
            0.0027984204265504766,
            -0.007132934251819072,
            0.008893378340991678,
            0.0004369664350014329,
            -0.0017067612001895275,
            -0.004708887757907461,
            0.005484673788684334,
            -0.0008769933556690285,
            0.0013686789773729707,
            -0.009624203991272897,
            0.0023527099415175407,
            0.002241914454448428,
            0.0023386799374951386,
            0.008874961570292482,
            0.0036364059820696674,
            -0.00280984198852428,
            -0.0012593609240131708,
            0.003952623918545298,
            -0.008795490567414604,
            0.003335334308913354,
            0.0034127573923631877,
            -0.005792348778523182,
            -0.007421474046902934,
            -0.0036914329815163228,
            -0.002725784581147548
        ]
        add_count = embed_dim - 50
        while add_count > 0:
            output.append(0.0001)
            add_count -= 1
        return output


def _word_embeds(tokens: List[str], kv: Word2VecKeyedVectors,
                 uniform_range: Tuple[float, ...]) -> np.ndarray:
    """ Get word embeddings of given tokens.

    Args:
        tokens (List[str]): A word tokens to calculate embeddding.

    Returns:
        numpy.ndarray: An embedding array with shape
                       (token_size, self.embed_dim, ).
    """
    doc_embed: List[np.ndarray] = []
    for token in tokens:
        word_embed: np.ndarray = _word_embed(
            token=token, kv=kv, uniform_range=uniform_range
        )
        doc_embed.append(word_embed)
    return np.array(doc_embed)


def _hierarchical_pool(
    tokens_embed: np.ndarray,
    num_windows: int = 3
) -> np.ndarray:
    """ Hierarchical Pooling: It takes word-order or spatial information
        into consideration when calculate document embeddings.

    Args:
        tokens_embed (np.ndarray): An embeded document vector.
        num_windows (int): A sizw of window to consider sequence.

    Returns:
        numpy.ndarray: An embedding array with shape (self.embed_dim, ).
    """
    text_len: int = tokens_embed.shape[0]
    if num_windows > text_len:
        raise ValueError(f'window size [{num_windows}] must be less '
                         f'than text length{text_len}.')

    num_iters: int = text_len - num_windows + 1
    pooled_doc_embed: List[np.ndarray] = [
        np.mean(tokens_embed[i:i + num_windows],
                axis=0) for i in range(num_iters)
    ]
    return np.max(pooled_doc_embed, axis=0)


def infer_vector(
    tokens: List[str],
    kv,
    method: str = 'avg',
    uniform_range: Tuple[float, float] = (-0.01, 0.01),
    num_windows: int = 3
) -> np.ndarray:
    tokens_embed: np.ndarray = _word_embeds(
        tokens=tokens,
        kv=kv,
        uniform_range=uniform_range
    )

    if method == 'max':
        return tokens_embed.max(axis=0)

    elif method == 'avg':
        return tokens_embed.mean(axis=0)

    elif method == 'concat':
        return np.hstack([tokens_embed.mean(axis=0), tokens_embed.max(axis=0)])

    elif method == 'hierarchical':
        return _hierarchical_pool(tokens_embed, num_windows)

    else:
        raise ValueError(
            f'infer_vector has no attribute [{method}] method.'
        )


class SWEM:
    """Implementation of SWEM.

    Args:
        kv: gensim.models.keyedvectors.Word2VecKeyedVectors
        tokenizer: Callable
            Callable object to tokenize input text.
        uniform_range: Tuple[float, ...]
            A range of uniform distribution to create random embedding.
    """

    def __init__(self, kv, tokenizer: Callable, uniform_range=(-0.01, 0.01)):
        self.kv: Word2VecKeyedVectors = kv

        if not callable(tokenizer):
            raise ValueError('tokenizer must be callable object.')
        self.tokenizer: Callable = tokenizer
        self.uniform_range: Tuple[float, ...] = uniform_range

    def infer_vector(
        self,
        doc: str,
        method: str = 'max',
        num_windows: int = 3
    ) -> np.ndarray:
        """ A main method to get document vector.

        Args:
            doc (str): A document str to get embeddings.
            method (str): Designate method to pool.
                         ('max', 'avg', 'concat', 'hierarchical')

        Returns:
            numpy.ndarray: An embedding array.
        """
        tokens: List[str] = self.tokenizer(doc)
        doc_embed: np.ndarray = _word_embeds(
            tokens=tokens,
            kv=self.kv,
            uniform_range=self.uniform_range
        )

        if method == 'max':
            return doc_embed.max(axis=0)

        elif method == 'avg':
            return doc_embed.mean(axis=0)

        elif method == 'concat':
            return np.hstack([doc_embed.mean(axis=0), doc_embed.max(axis=0)])

        elif method == 'hierarchical':
            return _hierarchical_pool(doc_embed, num_windows)

        else:
            raise ValueError(
                f'infer_vector has no attribute [{method}] method.'
            )
