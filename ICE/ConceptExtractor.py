import numpy as np
from sklearn.decomposition import NMF

class ConceptExtractor:
    def __init__(self, A, seed):
        self.seed = seed # seed for NMF
        self.batch_size, self.height, self.width, self.channels = A.shape
        self.A = A
        self.A_approx = None
        self.nr_concepts = None
        self.V = None
        self.V_approx = None  # V' = S*P
        self.S = None  # feature score, indicator matrix of which parts are present in the observations (n x h x w, c') --> where are the concepts
        self.P = None  # feature direction, vector basis, how the parts are characterized or measured by the observations (c', c) --> what concepts
        
        self.NMF_quality = None
        self.explainer_NMF_quality = None # get the Frobenius norm of the transformed A_to_explain
        self.nmf_model = None # fitted model (fitten in apply NMF) and uses the same P for transforming other activation layers

    def apply_NMF(self, nr_concepts): # euqals reducer in paper reducer
        if nr_concepts < 1:
            print("At least one concept must be specified")
            return
        # reshape A 4dim (batch_size, height, width, channels) --> 2dim (batch_size x height x width, channels)
        self.nr_concepts = nr_concepts
        self.V = self.A.reshape((self.batch_size * self.height * self.width, self.channels))

        # apply NMF to V --> reducer
        self.nmf_model = NMF(n_components=nr_concepts, random_state=self.seed, beta_loss='frobenius')
        self.S = self.nmf_model.fit_transform(self.V)
        self.P = self.nmf_model.components_
        self.V_approx = self.S @ self.P
        self.NMF_quality = self.calculate_NMF_quality(self.V, self.V_approx)
        self.A_approx = self.get_A_approx()
        
    def calculate_NMF_quality(self, V, V_approx):
        # use the Frobenius norm to calculate the error of NMF
        residual_error = V - V_approx
        return np.linalg.norm(residual_error.astype(np.float64), ord="fro")

    def get_A_approx(self):
        # reshape V' to get A' which is the approximated activation layer
        A_approx = self.V_approx.reshape((self.batch_size, self.height, self.width, self.channels))
        return A_approx

    def explainer(self, A_to_explain, batch_size=None):
        # explain what concepts (are in P) are present/ where these concepts are in the new activation layer A_to_explain
        # note: batch size can be different but the rest should be the same
        # calculate S given A_to_explain and P
        if A_to_explain.shape[3] != self.A.shape[3]:
            print(f"A_to_explain must have the same shape as A used for NMF fitting!")
            # otherwise nmf_model.transform(V_to_explain) throws an error
            return
        if batch_size is None:
            batch_size = self.batch_size
        V_to_explain = A_to_explain.reshape(batch_size * self.height * self.width, self.channels)  # reuse the shapes as they must be the same, except for the batch size
        S_to_explain = self.nmf_model.transform(V_to_explain)  # use the fitted model from above to get S_to_explain for our new A_to_explain (P stays the same)
        self.explainer_NMF_quality = self.calculate_NMF_quality(V_to_explain, S_to_explain @ self.P)
        return S_to_explain
    