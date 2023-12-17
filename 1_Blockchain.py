import hashlib
import json
import datetime
import streamlit as st
import pandas as pd
from deta import Deta

# Initialize Deta and create a database instance
deta = Deta('d0fffmsczxt_6eGtX2hUDMSnDts3vhQLdwuTSePG2d4x')
db = deta.Base('blocks')

class Block:
    def __init__(self, index, timestamp, data, previous_hash):
        self.index = index
        self.timestamp = timestamp
        self.data = data
        self.previous_hash = previous_hash
        self.hash = self.calculate_hash()

    def calculate_hash(self):
        data = str(self.index) + str(self.timestamp) + str(self.data) + str(self.previous_hash)
        encoded_bytes = data.encode('utf-8')
        hashed = hashlib.sha256(encoded_bytes).hexdigest()
        return hashed

class Blockchain:
    def __init__(self):
        self.chain = self.load_chain()

    def load_chain(self):
        all_blocks = db.fetch().items
        if not all_blocks:
            return [self.create_genesis_block()]
        else:
            blockchain = []
            for block_data in all_blocks:
                index = block_data['index']
                timestamp = datetime.datetime.strptime(block_data['timestamp'], "%Y-%m-%d %H:%M:%S")
                data = block_data['data']
                previous_hash = block_data['previous_hash']
                block = Block(index, timestamp, data, previous_hash)
                blockchain.append(block)
            return blockchain

    def create_genesis_block(self):
        return Block(0, datetime.datetime.now(), "Genesis Block", "0")

    def get_latest_block(self):
        return self.chain[-1]

    def add_block(self, new_block):
        new_block.previous_hash = self.get_latest_block().hash
        new_block.hash = new_block.calculate_hash()
        self.chain.append(new_block)
        timestamp_str = new_block.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        data = {
            'index': new_block.index,
            'timestamp': timestamp_str,
            'data': new_block.data,
            'previous_hash': new_block.previous_hash,
            'hash': new_block.hash
        }
        db.put(data)

    def is_chain_valid(self):
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i - 1]
            if current_block.hash != current_block.calculate_hash() or \
               current_block.previous_hash != previous_block.hash:
                return False
        return True

def registration_page():
    st.title(":safety_pin: Blockchain Registration")

    blockchain = Blockchain()

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    email = st.text_input("Email")

    # Display blockchain information
    st.header("Blockchain Benefits:")
    st.markdown("- **Security**: Blockchain uses cryptographic techniques to secure data.")
    st.markdown("- **Transparency**: All transactions are recorded and can be audited.")
    st.markdown("- **Immutability**: Once data is recorded, it cannot be easily altered.")
    st.markdown("- **Decentralization**: No single authority controls the blockchain.")
    
    if st.button("Register"):
        # Implement user registration with public-key cryptography
        # Store user data in the blockchain
        user_data = {
            'username': username,
            'email': email,
            'public_key': 'd0fffmsczxt_6eGtX2hUDMSnDts3vhQLdwuTSePG2d4x',  # Replace with actual public key
            'registration_time': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        new_block = Block(len(blockchain.chain), datetime.datetime.now(), json.dumps(user_data), "")
        blockchain.add_block(new_block)
        st.success("Registration successful! Please log in.")
        return True

def login_page():
    st.title("Blockchain Login")

    blockchain = Blockchain()

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    # Display blockchain information
    st.header("Blockchain Security Features:")
    st.markdown("- **Cryptography**: Data is hashed and secured using cryptographic algorithms.")
    st.markdown("- **Consensus**: Transactions are verified by network participants.")
    st.markdown("- **Immutable Ledger**: Once data is added, it cannot be deleted or modified.")
    
    if st.button("Login"):
        if blockchain.is_chain_valid():
            all_blocks = db.fetch().items
            for block in all_blocks:
                user_data = json.loads(block['data'])
                if user_data.get('username') == username:
                    # Implement authentication using the user's public key and password
                    st.success("Login successful!")
                    return True
            else:
                st.error("Invalid username or password")
        else:
            st.error("Blockchain integrity compromised")

def transparency_page():
    st.title("Blockchain Transparency")
    all_blocks = db.fetch().items
    if all_blocks:
        df = pd.DataFrame(all_blocks)
        st.dataframe(df, height=500)
    else:
        st.info("No records found in the blockchain.")

def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Select a page", ["Register", "Login", "Transparency"])
    if page == "Register":
        if registration_page():
            st.sidebar.text("You can now log in.")
    elif page == "Login":
        if login_page():
            st.sidebar.text("Welcome!")
    elif page == "Transparency":
        transparency_page()

if __name__ == "__main__":
    main()
