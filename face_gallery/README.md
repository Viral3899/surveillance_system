# Face Gallery

This directory contains known face images for recognition.

## Adding Face Images:

1. **One face per image**: Each image should contain only one person's face
2. **Good quality**: Use clear, well-lit photos  
3. **Naming convention**: Use the person's name as the filename
   - Example: `john_doe.jpg`, `jane_smith.png`
4. **Supported formats**: JPG, JPEG, PNG, BMP

## Examples:

```
face_gallery/
├── alice_johnson.jpg
├── bob_wilson.png
├── charlie_brown.jpg
└── diana_prince.jpeg
```

## Tips:

- Use multiple photos of the same person with different angles/lighting
- Name them: `alice_01.jpg`, `alice_02.jpg`, etc.
- Avoid group photos or multiple faces in one image
- Face should be clearly visible and not obscured

The system will automatically process these images and create face encodings for recognition.
