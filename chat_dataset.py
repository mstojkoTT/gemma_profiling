import random
import os

picture_dir = os.path.join(os.getcwd(), "pictures")

def get_random_message():
    img_idx = random.randint(0, 9)
    chat_idx = random.randint(0, 9)

    ret = messages_list[chat_idx]
    ret[1]["content"][0]["image"] = os.path.join(picture_dir, f"picture{img_idx}.jpg")

    return ret


messages_list = [
    [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant."}]
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "/home/ttuser/mstojko/gemma_profiling/picture0.jpg"},
                {"type": "text", "text": "Explain all the visual elements and objects visible in this image."}
            ]
        }
    ],
    [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant."}]
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "/home/ttuser/mstojko/gemma_profiling/picture1.jpg"},
                {"type": "text", "text": "Provide a thorough description of what is shown in this image."}
            ]
        }
    ],
    [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant."}]
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "/home/ttuser/mstojko/gemma_profiling/picture2.jpg"},
                {"type": "text", "text": "Describe all the key features and elements seen in this image."}
            ]
        }
    ],
    [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant."}]
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "/home/ttuser/mstojko/gemma_profiling/picture3.jpg"},
                {"type": "text", "text": "Explain the scene depicted in the image with precise details."}
            ]
        }
    ],
    [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant."}]
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "/home/ttuser/mstojko/gemma_profiling/picture4.jpg"},
                {"type": "text", "text": "Give a detailed breakdown of everything shown in this picture."}
            ]
        }
    ],
    [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant."}]
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "/home/ttuser/mstojko/gemma_profiling/picture5.jpg"},
                {"type": "text", "text": "Describe the visual composition and all items in this image."}
            ]
        }
    ],
    [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant."}]
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "/home/ttuser/mstojko/gemma_profiling/picture6.jpg"},
                {"type": "text", "text": "Explain the contents and notable aspects of this image."}
            ]
        }
    ],
    [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant."}]
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "/home/ttuser/mstojko/gemma_profiling/picture7.jpg"},
                {"type": "text", "text": "Provide a clear and thorough description of this scene."}
            ]
        }
    ],
    [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant."}]
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "/home/ttuser/mstojko/gemma_profiling/picture8.jpg"},
                {"type": "text", "text": "Detail all objects and characteristics visible in this image."}
            ]
        }
    ],
    [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant."}]
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "/home/ttuser/mstojko/gemma_profiling/picture9.jpg"},
                {"type": "text", "text": "Describe every important visual detail present in this picture."}
            ]
        }
    ]
]
