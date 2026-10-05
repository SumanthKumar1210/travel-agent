from agent import TravelAgent
from load_dotenv import load_dotenv
import argparse

if __name__ == "__main__":
    # Create argument parser
    parser = argparse.ArgumentParser(description='Run the Travel Agent')
    parser.add_argument('--model', type=str, default=None,
                        help='Optional model override for the Supervisor (e.g. "openai:gpt-4o"). Worker agents keep their own configured models. Defaults to SUPERVISOR_MODEL / agent/supervisor.py.')
    parser.add_argument('--query', type=str, required=True,
                        help='The travel query to process')
    parser.add_argument('--env', type=str, default=".env",
                        help='Path to the environment file containing API keys')
    
    
    # Parse arguments
    args = parser.parse_args()

    env_loaded = load_dotenv(args.env)
    if not env_loaded:
        raise ValueError(f"Failed to load environment variables from {args.env}")
    
    
    # Create agent with specified model
    agent = TravelAgent()
    
    # Run the query
    output, trace = agent.run(args.query, args.model)
    print(output)
