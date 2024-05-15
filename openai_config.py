import openai


class OpenAIConfig:
    def __init__(
        self,
        api_key: str = '',
        base_url: str = 'https://api.openai.com/v1',
        model: str = 'gpt-4',
        max_tokens: int = 8192,
        stream: bool = False,
        temperature: float = 0.5,
        top_p: float = 1.0,
        timeout: int = 10,
        system_message: str = 'You are a helpful assistant.',
    ):
        self.api_key = api_key
        self.old_api_key = api_key
        self.base_url = base_url
        self.old_base_url = base_url
        self.model = model
        self.max_tokens = max_tokens
        self.stream = stream
        self.temperature = temperature
        self.top_p = top_p
        self.timeout = timeout
        self.system_message = system_message

        self._client = self.get_client(renew=True)

    def get_client(self, renew: bool = False) -> openai.OpenAI:
        if self.api_key != self.old_api_key or self.base_url != self.old_base_url or renew:
            self.old_api_key = self.api_key
            self.old_base_url = self.base_url
            self._client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)

        return self._client

    def get_models(self) -> list:
        models_data = self.get_client().models.list().model_dump()['data']
        models_list = [model['id'] for model in models_data]
        return models_list

    def openai_chat_completion(self, messages):
        response = self.get_client().chat.completions.create(
            messages=messages, model=self.model, max_tokens=self.max_tokens, stream=self.stream, temperature=self.temperature, top_p=self.top_p, timeout=self.timeout
        )

        return response

    def chat(self, content: str):
        messages = [{'role': 'system', 'content': self.system_message}, {'role': 'user', 'content': content}]
        response = self.openai_chat_completion(messages)
        return response.choices[0].message.content
