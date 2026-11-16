"""
deploy/create_user.py
---------------------
Creates an account from the command line - the way to add people when
SIGNUP_MODE=closed. Run it on the server, inside the container if Docker:

    python deploy/create_user.py "Dr. A. Deshmukh" mentor@example.com
    docker exec -it sentiment-studio python deploy/create_user.py "Name" email@example.com

It asks for the password without showing it. Uses ACCOUNTS_DIR like the app.
"""
import getpass
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import auth  # noqa: E402


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    name, email = sys.argv[1], sys.argv[2]
    password = getpass.getpass(f"Password for {email} (8+ characters): ")
    if password != getpass.getpass("Repeat it: "):
        sys.exit("The passwords don't match.")
    try:
        user = auth.create_user(name, email, password)
    except auth.AuthError as e:
        sys.exit(f"{e.title} {e.hint}".strip())
    print(f"Created {user['email']} in {auth.ACCOUNTS_DIR}")


if __name__ == "__main__":
    main()
