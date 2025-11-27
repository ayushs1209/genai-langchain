from langchain_google_genai import ChatGoogleGenerativeAI
from typing_extensions import TypedDict, Annotated
from dotenv import load_dotenv

load_dotenv()

SYSTEM = """
    you are a customer satisfaction chatbot, whose job is summarize the user feedback in short and then return as the output with sentiment
"""

model = ChatGoogleGenerativeAI(model = "gemini-2.5-flash")


class outputSchema(TypedDict):
    """way to return the output. always keep the summary key first then the sentiment key"""
    summary: Annotated[str, ..., "summary of the user feedback review"]
    sentiment: Annotated[str, ..., "return the sentiment of the user review"]


structuredModel = model.with_structured_output(outputSchema)

userFeedback = [
    "I was pleasantly surprised by how well this device performed right out of the box. The setup process took only a few minutes, and the interface felt intuitive and polished. After a full day of use, the battery life remained impressive, and the overall build quality felt sturdy and reliable for everyday tasks.",
    
    "My experience with this product was far from ideal. The material felt cheap the moment I picked it up, and several features didn't work as advertised. After a few hours of use, it began to overheat and slow down noticeably. For the price I paid, the quality and performance were seriously disappointing.",
    
    "Overall, the product was neither great nor terrible—it simply did what it was supposed to do. The performance was consistent, the controls were straightforward, and nothing major went wrong during use. It's not something I'd rave about, but it's decent enough for someone looking for a basic, functional solution.",
    
    "This exceeded my expectations by a huge margin. The design is sleek, the performance is incredibly fast, and the user experience feels thoughtfully crafted. Even when multitasking heavily, it handled everything without slowing down. It's easily one of the best purchases I've made this year and offers excellent long-term value.",
    
    "The delivery was quick, but the product itself felt underwhelming. While it worked fine for the most part, a few features behaved unpredictably, and the overall design lacked refinement. It's usable, but you can tell corners were cut in manufacturing. It's acceptable for light use but not ideal for demanding tasks."
]


for message in userFeedback :
    result =  structuredModel.invoke([
    [
        "system",
        SYSTEM,
    ],
    ["human", message],
])
    print (result)
