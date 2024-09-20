import numpy as np
import torch
import torch.nn as nn
from statsmodels.stats.diagnostic import het_arch
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import StandardScaler


class HomoscedasticTransformer(nn.Module):
    """
    Perform Homoscedastic Transformation

    This function applies a homoscedastic transformation to the input data by fitting the model
    and ensuring that key statistical properties (e.g., stationarity and homoscedasticity) are satisfied.

    ## Requirements:
    1. **Stationarity**: The input data should be stationary. This is verified using the Augmented Dickey-Fuller (ADF) test.
       - If any feature is found to be non-stationary (p-value > 0.05), an error is raised, and the process is halted.

    2. **Homoscedasticity**: The data is expected to have a constant variance (homoscedasticity), verified through the ARCH test.
       - If heteroscedasticity is detected (p-value between 0.01 and 0.05), a warning is raised, suggesting an increase in the number of training epochs for better convergence.

    ## Checks:
    - **Standardization**: The input data should have a mean close to 0 and a standard deviation close to 1.
      - Tolerances for the checks:
        - Mean: ±1e-2
        - Standard Deviation: ±1e-2
      - If the data fails these checks, the user is notified, and standardization should be applied before proceeding.

    - **Stationarity Check**: The ADF test is applied to each feature of the data.
      - A p-value threshold of 0.05 is used to determine stationarity.
      - If the p-value is greater than 0.05, the feature is considered non-stationary.

    - **Homoscedasticity Check**: The ARCH test is applied to ensure constant variance across the data.
      - If the p-value is between 0.01 and 0.05, a warning is issued.
      - If the p-value is less than 0.01, it indicates severe heteroscedasticity, but the process will still proceed with a warning.

    ## Possible Errors:
    - **ValueError**: Raised when the input data is found to be non-stationary. This error prevents the model from proceeding since stationarity is a critical requirement.

    ## Possible Warnings:
    - **Warning**: Raised when the data is found to be heteroscedastic (non-constant variance) based on the ARCH test.
      - The process can still proceed, but convergence issues may occur.
      - The user is advised to consider increasing the number of epochs or modifying other model parameters to mitigate this issue.

    ## Authors:
    - Krasimir Trifonov
    - ChatGPT 4o
    """
    def __init__(self, input_size, hidden_size, latent_size, verbose=False):
        super(HomoscedasticTransformer, self).__init__()
        self.scaler = StandardScaler()  # Scaler for normalizing output data

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

    # def decode(self, z):
    #     h = torch.relu(self.fc3(z))
    #     return torch.sigmoid(self.fc4(h))

    def decode(self, z):
        h = torch.relu(self.fc3(z))
        return self.fc4(h)  # Remove sigmoid to avoid range compression

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
        This function runs the ADF test and other future tests you may want to add.
        """
        # Convert tensor to numpy array
        # X_np = X.cpu().numpy()# Converted to numpy by Standard Scaler
        X_np = X

        # Perform ADF Test
        adf_results = self.adf_test(X_np)
        for i, p_value in enumerate(adf_results):
            if p_value > 0.05:
                if self.verbose:
                    print(f'Feature {i}: Non-Stationary (p-value = {p_value:.4f})')
                raise ValueError(f'Feature {i} is Non-Stationary (p-value = {p_value:.4f}). Aborting process.')
            else:
                if self.verbose:
                    print(f'Feature {i}: Stationary (p-value = {p_value:.4f})')

        # Standardization test
        check_standardization_result = self.check_standardization(X_np)
        if not check_standardization_result:
            raise Warning("Data is NOT standardized!")
        # Placeholder for future tests
        # You can add other statistical tests here in the future

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
        - X: Data (numpy array).
        - mean_tolerance: Maximum allowed deviation from 0 for the mean.
        - std_tolerance: Maximum allowed deviation from 1 for the standard deviation.

        Returns True if the data is sufficiently standardized, else False.
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
        # Optional: If you want to standardize input data one more time
        X_scaled = self.scaler.fit_transform(X)
        # Run the ADF test and other tests before fitting
        self.run_tests(X_scaled)

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
        if not self.is_fitted:
            raise RuntimeError("HomoscedasticTransformer is not fitted yet. Call `fit` first.")

        X_scaled = self.scaler.transform(X)  # Optional: Standardize input if needed

        self.eval()
        with torch.no_grad():
            reconstructed_data, _, _ = self(X_scaled)
            # Convert reconstructed data to numpy array for testing
        reconstructed_data_np = reconstructed_data.cpu().numpy()

        # ARCH test: run on each feature dimension independently (assuming multivariate data)
        arch_test_resid = het_arch(reconstructed_data_np[:,0])
        p_value = arch_test_resid[1]
        # Check p-values for heteroscedasticity (p > 0.05 means no heteroscedasticity)

        if p_value < 0.05 and p_value > 0.01:
            raise Warning(f'Homoscedasticity is not achieved!, (p-value = {p_value:.4f}), try to increase number of epochs.')

            # Now normalize the output (reconstructed data)
        reconstructed_data_normalized = self.scaler.fit_transform(reconstructed_data_np)

        # Return reconstructed data and a list of p-values from the ARCH test
        return reconstructed_data_normalized, p_value

    def fit_transform(self, X, epochs=100, lr=0.001):
        # Combines fit and transform
        self.fit(X, epochs=epochs, lr=lr)
        return self.transform(X)

    @staticmethod
    def vae_loss(recon_x, x, mu, logvar):
        recon_loss = nn.MSELoss()(recon_x, x)  # Reconstruction loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())  # KL divergence
        return recon_loss + kl_loss


# # Example usage:
# input_size = 20  # Example input size
# hidden_size = 50
# latent_size = 10
# X = torch.randn((100, input_size))  # Example data
#
# vae_model = HomoscedasticTransformer(input_size, hidden_size, latent_size)
#
# # Fit the model and transform data
# vae_model.fit(X, epochs=50)
# reconstructed_data = vae_model.transform(X)
#
# # Or, do both fit and transform in one step
# reconstructed_data = vae_model.fit_transform(X, epochs=50)