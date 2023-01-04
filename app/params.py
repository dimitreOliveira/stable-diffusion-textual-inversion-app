EPOCHS: int = 1  # 50
MAX_PROMPT_LENGTH: int = 77
placeholder_token: str = "<custom-token>"
initializer_token: str = "cat"
output_folder: str = "example"

single_urls: list[str] = [
    "https://i.imgur.com/VIedH1X.jpg",
    "https://i.imgur.com/eBw13hE.png",
    "https://i.imgur.com/oJ3rSg7.png",
    "https://i.imgur.com/5mCL6Df.jpg",
    "https://i.imgur.com/4Q6WWyI.jpg",
]

group_urls: list[str] = [
    "https://i.imgur.com/yVmZ2Qa.jpg",
    "https://i.imgur.com/JbyFbZJ.jpg",
    "https://i.imgur.com/CCubd3q.jpg",
]

single_prompts: list[str] = [
    "a photo of a {}",
    "a rendering of a {}",
    "a cropped photo of the {}",
    "the photo of a {}",
    "a photo of a clean {}",
    "a photo of my {}",
    "a photo of the cool {}",
    "a close-up photo of a {}",
    "a bright photo of the {}",
    "a cropped photo of a {}",
    "a photo of the {}",
    "a good photo of the {}",
    "a photo of one {}",
    "a close-up photo of the {}",
    "a rendition of the {}",
    "a photo of the clean {}",
    "a rendition of a {}",
    "a photo of a nice {}",
    "a good photo of a {}",
    "a photo of the nice {}",
    "a photo of the small {}",
    "a photo of the weird {}",
    "a photo of the large {}",
    "a photo of a cool {}",
    "a photo of a small {}",
]

group_prompts: list[str] = [
    "a photo of a group of {}",
    "a rendering of a group of {}",
    "a cropped photo of the group of {}",
    "the photo of a group of {}",
    "a photo of a clean group of {}",
    "a photo of my group of {}",
    "a photo of a cool group of {}",
    "a close-up photo of a group of {}",
    "a bright photo of the group of {}",
    "a cropped photo of a group of {}",
    "a photo of the group of {}",
    "a good photo of the group of {}",
    "a photo of one group of {}",
    "a close-up photo of the group of {}",
    "a rendition of the group of {}",
    "a photo of the clean group of {}",
    "a rendition of a group of {}",
    "a photo of a nice group of {}",
    "a good photo of a group of {}",
    "a photo of the nice group of {}",
    "a photo of the small group of {}",
    "a photo of the weird group of {}",
    "a photo of the large group of {}",
    "a photo of a cool group of {}",
    "a photo of a small group of {}",
]
