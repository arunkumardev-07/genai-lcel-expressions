## Design and Implementation of LangChain Expression Language (LCEL) Expressions

### AIM:
To design and implement a LangChain Expression Language (LCEL) expression that utilizes at least two prompt parameters and three key components (prompt, model, and output parser), and to evaluate its functionality by analyzing relevant examples of its application in real-world scenarios.

### PROBLEM STATEMENT:

### DESIGN STEPS:
### STEP 1: Setup API and Environment: Load environment variables using dotenv and set openai.api_key from the local environment.
### STEP 2: Create Prompt and Model: Use LangChain to define a ChatPromptTemplate and initialize ChatOpenAI for text generation.
### STEP 3: Build a Retrieval System: Store predefined texts in DocArrayInMemorySearch with OpenAIEmbeddings and create a retriever.
### STEP 4: Define Question-Answering Chain: Use RunnableMap to fetch relevant documents and pass them to a chat model for responses.
### STEP 5: Invoke the Chain: Run chain.invoke() with a question to retrieve context-based answers using the LangChain pipeline.

### PROGRAM:
```


from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import ResponseSchema
from langchain.schema import OutputParserException
from langchain.schema.output_parser import StrOutputParser


# In[2]:


# Define the PromptTemplate
prompt = PromptTemplate(
    template="""
You are a travel assistant. Based on the following inputs, recommend a destination:
- Preferred activity: {activity}
- Budget (in USD): {budget}

Provide a response strictly in JSON format:
{{
    "destination": "<destination>",
    "activity": "<activity>",
    "cost": "<cost>"
}}
""",
    input_variables=["activity", "budget"],
)


# In[3]:


# Define the Output Parser
response_schemas = [
    ResponseSchema(name="destination", description="Recommended travel destination"),
    ResponseSchema(name="activity", description="Suggested activity at the destination"),
    ResponseSchema(name="cost", description="Estimated cost in USD for the trip"),
]
output_parser = StrOutputParser()


# In[4]:


# Initialize the LLM
llm = ChatOpenAI(model="gpt-4-0613", temperature=0)


# In[5]:


# Create the LangChain Expression (LLM Chain)
chain = LLMChain(llm=llm, prompt=prompt, output_parser=output_parser)


# In[6]:


# Test the chain with an example
input_data = {"activity": "hiking", "budget": 1000}
result = chain.run(input_data)


# In[7]:


# Parse the structured output
parsed_result = output_parser.parse(result)


# In[8]:


# Display the result
print("Recommendation:", parsed_result)


```





### OUTPUT:
![image](https://github.com/user-attachments/assets/be5106b6-4de8-4473-bd35-a6d7016b5089)


### RESULT:
The implemented LCEL expression takes at least two prompt parameters, processes them using a model, and formats the output with a parser, demonstrating its effectiveness through real-world examples.
