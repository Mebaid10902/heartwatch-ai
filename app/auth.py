from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import HTTPBasic, HTTPBasicCredentials, OAuth2PasswordBearer, OAuth2PasswordRequestForm
import secrets
import jwt
import time

router = APIRouter()
security = HTTPBasic()

# JWT Configuration
SECRET_KEY = "your_secret_key"
ALGORITHM = "HS256"

# Dummy user database (for demo only)
USERS = {
    "admin": "password123",  # In production, use hashed passwords
}

# Function to create a JWT token
def create_access_token(data: dict, expires_delta: int = 3600):
    to_encode = data.copy()
    expire = time.time() + expires_delta
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# For /secure using HTTP Basic Auth
def verify_user(credentials: HTTPBasicCredentials = Depends(security)):
    correct_username = secrets.compare_digest(credentials.username, "admin")
    correct_password = secrets.compare_digest(credentials.password, USERS["admin"])
    if not (correct_username and correct_password):
        raise HTTPException(status_code=401, detail="Unauthorized")
    return credentials.username

@router.get("/secure")
def secure_data(user: str = Depends(verify_user)):
    return {"message": f"Hello, {user}. You're authorized via HTTP Basic."}

# OAuth2 setup
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/login")

# JWT-based token login endpoint
@router.post("/login")
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = form_data.username
    password = form_data.password
    if user in USERS and secrets.compare_digest(password, USERS[user]):
        token = create_access_token({"sub": user})
        return {"access_token": token, "token_type": "bearer"}
    raise HTTPException(status_code=401, detail="Invalid credentials")

# Token verification function
def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return username
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

# JWT-protected route
@router.get("/protected")
def protected_route(user: str = Depends(get_current_user)):
    return {"message": f"Welcome, {user}. You accessed a token-protected route."}
