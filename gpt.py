from openai import OpenAI
import os
import json

def load_api_key():
    """
    Load OpenAI API key from environment variable or config file.
    Returns the API key as a string or None if not found.
    """
    api_key = os.environ.get('OPENAI_API_KEY')
    
    if not api_key:
        config_path = os.path.join(os.path.dirname(__file__), '.env')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                for line in f:
                    if line.startswith('OPENAI_API_KEY='):
                        api_key = line.strip().split('=', 1)[1].strip('"\'')
                        break
    
    return api_key

api_key = load_api_key()
if api_key:
    os.environ['OPENAI_API_KEY'] = api_key

class GPTResponse:
    def get_response(self, prompt_text, schema=None):
        """
        Get response from GPT model.
        
        Args:
            prompt_text (str): The prompt text to send to GPT
            schema (dict, optional): JSON schema for structured output. If provided, 
                                   the response will be structured according to this schema.
                                   Example:
                                   {
                                       "type": "object",
                                       "properties": {
                                           "is_duplicate": {"type": "boolean"},
                                           "keep_item": {"type": "string", "enum": ["1", "2", "0"]},
                                           "reason": {"type": "string"}
                                       },
                                       "required": ["is_duplicate", "keep_item", "reason"],
                                       "additionalProperties": False
                                   }
        
        Returns:
            If schema is provided: dict containing structured response
            If no schema: str containing the raw response
        """
        try:
            # Check if API key is set
            if not os.environ.get('OPENAI_API_KEY'):
                raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable or create a .env file.")
                
            client = OpenAI()
            
            if schema:
                # Create structured output request
                response = client.responses.create(
                    model="gpt-4o-mini",
                    input=[
                        {"role": "system", "content": "You are a helpful assistant good at history and bibliographic analysis."},
                        {"role": "user", "content": prompt_text}
                    ],
                    text={
                        "format": {
                            "type": "json_schema",
                            "name": "response",
                            "schema": schema,
                            "strict": True
                        }
                    }
                )
                return json.loads(response.output_text)
            else:
                # Create regular chat completion
                completion = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant good at history."},
                        {"role": "user", "content": prompt_text}
                    ]
                )
                return completion.choices[0].message.content.strip()
                
        except Exception as e:
            print(f"Error in GPT response: {e}")
            return None

if __name__ == "__main__":
    gpt_generate_topics = GPTResponse()
    topic = "武谷三男の思想形成(1938年の検挙まで)"
    prompt_text = f"""Translate this title ("{topic}") into English. Your answer should strictly follow this example: ["translated text"]. No explanation."""
    print(gpt_generate_topics.get_response(prompt_text))
