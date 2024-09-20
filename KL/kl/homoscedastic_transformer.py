import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import adfuller
import random

def set_seed(seed):
    """
    Set seeds for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False




class HomoscedasticTransformer(nn.Module):
    """
    Perform Homoscedastic Transformation with StandardScaler normalization for input and output.
    """

    def __init__(self, input_size, hidden_size, latent_size, verbose=False):
        super(HomoscedasticTransformer, self).__init__()
        set_seed(42) # For Reproducibility
        self.scaler = StandardScaler()  # Scaler for normalizing input and output data
        self.verbose = verbose

        # Encoder
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc_mu = nn.Linear(hidden_size, latent_size)  # Mean of latent space
        self.fc_logvar = nn.Linear(hidden_size, latent_size)  # Log variance of latent space

        # Decoder
        self.fc3 = nn.Linear(latent_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, input_size)

        # To keep track if the model is fitted
        self.is_fitted = False

    def encode(self, x):
        h = torch.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def decode(self, z):
        h = torch.relu(self.fc3(z))
        return self.fc4(h)  # No sigmoid to avoid range compression

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std  # Sample from latent space

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def run_tests(self, X):
        """
        This function runs the ADF test and standardization checks on the input data.
        """
        # Perform ADF Test
        adf_results = self.adf_test(X)
        for i, p_value in enumerate(adf_results):
            if p_value > 0.05:
                if self.verbose:
                    print(f'Feature {i}: Non-Stationary (p-value = {p_value:.4f})')
                raise ValueError(f'Feature {i} is Non-Stationary (p-value = {p_value:.4f}). Aborting process.')
            else:
                if self.verbose:
                    print(f'Feature {i}: Stationary (p-value = {p_value:.4f})')

        # Standardization test
        check_standardization_result = self.check_standardization(X)
        if not check_standardization_result:
            raise Warning("Data is NOT standardized!")

    def adf_test(self, X):
        """
        Performs the Augmented Dickey-Fuller (ADF) test for stationarity on each feature of X.
        """
        adf_results = []
        for i in range(X.shape[1]):
            adf_result = adfuller(X[:, i])
            adf_results.append(adf_result[1])  # Append p-value
        return adf_results

    def check_standardization(self, X, mean_tolerance=1e-2, std_tolerance=1e-2):
        """
        Checks if the data is standardized within a given tolerance.
        """
        mean = np.mean(X)
        std = np.std(X)
        variance = np.var(X)

        mean_check = np.abs(mean) <= mean_tolerance
        std_check = np.abs(std - 1) <= std_tolerance
        variance_check = np.abs(variance - 1) <= std_tolerance  # Variance is std^2, so same tolerance

        if mean_check and std_check and variance_check:
            if self.verbose:
                print(f"Data is standardized: Mean = {mean:.4f}, Std = {std:.4f}, Variance = {variance:.4f}")
            return True
        else:
            if self.verbose:
                print(f"Data is NOT standardized: Mean = {mean:.4f}, Std = {std:.4f}, Variance = {variance:.4f}")
            return False

    def fit(self, X, epochs=100, lr=0.001):
        """
        Fit the model on the standardized input data.
        """
        # Convert tensor to numpy for StandardScaler
        if isinstance(X, torch.Tensor):
            X_np = X.cpu().numpy()

        # Standardize the input data
        X_scaled_np = self.scaler.fit_transform(X_np)

        # Convert back to tensor after scaling
        X_scaled = torch.tensor(X_scaled_np, dtype=torch.float32)

        # Run the ADF test and other tests before fitting
        self.run_tests(X_scaled_np)

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        # Training loop
        for epoch in range(epochs):
            self.train()
            recon_x, mu, logvar = self(X_scaled)
            loss = self.vae_loss(recon_x, X_scaled, mu, logvar)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0 and self.verbose:
                print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}, Variance loss:{torch.var(recon_x):.6f}')

        self.is_fitted = True  # Mark the model as fitted

    def transform(self, X):
        """
        Transform the input data using the fitted model and return normalized reconstructed data.
        """
        if not self.is_fitted:
            raise RuntimeError("HomoscedasticTransformer is not fitted yet. Call `fit` first.")

        # Convert input tensor to numpy for StandardScaler
        if isinstance(X, torch.Tensor):
            X_np = X.cpu().numpy()

        # Standardize input data
        X_scaled_np = self.scaler.transform(X_np)
        X_scaled = torch.tensor(X_scaled_np, dtype=torch.float32)  # Convert back to tensor

        self.eval()
        with torch.no_grad():
            reconstructed_data, _, _ = self(X_scaled)

        # Convert reconstructed data to numpy
        reconstructed_data_np = reconstructed_data.cpu().numpy()

        # Normalize the output (reconstructed data)
        reconstructed_data_normalized = self.scaler.fit_transform(reconstructed_data_np)

        return reconstructed_data_normalized

    def fit_transform(self, X, epochs=100, lr=0.001):
        """
        Fit the model and transform the input data, returning normalized reconstructed data.
        """
        self.fit(X, epochs=epochs, lr=lr)
        return self.transform(X)

    @staticmethod
    def vae_loss(recon_x, x, mu, logvar):
        """
        Compute the loss function for the VAE.
        """
        recon_loss = nn.MSELoss()(recon_x, x)  # Reconstruction loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())  # KL divergence
        return recon_loss + kl_loss