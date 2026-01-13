import os
from dotenv import load_dotenv

load_dotenv()

# Database
DATABASE_URL = os.getenv("DATABASE_URL", "mariadb+mariadbconnector://root:password@localhost:3306/pupero_moderation")

# Service port
MODERATION_PORT = int(os.getenv("MODERATION_PORT", "8020"))

# JWT Configuration (should match LoginBackend)
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-here")
JWT_ALGORITHM = "HS256"

# Service URLs for integration
LOGIN_SERVICE_URL = os.getenv("LOGIN_SERVICE_URL", "http://pupero-login:8001")
OFFERS_SERVICE_URL = os.getenv("OFFERS_SERVICE_URL", "http://pupero-offers:8002")
WALLET_SERVICE_URL = os.getenv("WALLET_SERVICE_URL", "http://pupero-WalletManager:8004")
TRANSACTIONS_SERVICE_URL = os.getenv("TRANSACTIONS_SERVICE_URL", "http://pupero-transactions:8003")

# Matrix integration
MATRIX_HS_URL = os.getenv("MATRIX_HS_URL", "http://pupero-matrix-synapse:8008")
MATRIX_ADMIN_SECRET = os.getenv("MATRIX_ADMIN_SECRET", "dev-shared-secret-change-me")
MATRIX_SERVER_NAME = os.getenv("MATRIX_SERVER_NAME", "Pupero")

# Logging
LOG_FILE = os.getenv("LOG_FILE", "/var/log/pupero/moderation.log")

# Moderation settings
AUTO_ESCALATION_THRESHOLD = int(os.getenv("AUTO_ESCALATION_THRESHOLD", "3"))  # Number of reports before auto-escalation
APPEAL_WINDOW_DAYS = int(os.getenv("APPEAL_WINDOW_DAYS", "30"))  # Days allowed to appeal a decision
