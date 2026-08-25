import os
from azure.ai.vision.face import FaceAdministrationClient
from azure.core.credentials import AzureKeyCredential
from config import AZURE_ENDPOINT, AZURE_KEY, PERSON_GROUP_ID

admin_client = FaceAdministrationClient(AZURE_ENDPOINT, AzureKeyCredential(AZURE_KEY))

def setup_and_train():
    try:
        # 1. Create PersonGroup if it doesn't exist
        print("Creating PersonGroup...")
        admin_client.person_group.create(
            person_group_id=PERSON_GROUP_ID,
            name="Employee Attendance Group"
        )
    except Exception as e:
        print(f"Note: PersonGroup might already exist ({e})")

    # 2. Add an Employee (Example: John Doe)
    print("Creating Employee record...")
    person = admin_client.person_group.create_person(
        person_group_id=PERSON_GROUP_ID,
        name="John Doe"
    )
    print(f"Created Person: {person.name} with ID: {person.person_id}")

    # 3. Add employee face image (Ensure sample_john.jpg exists in your project folder)
    image_path = "sample_john.jpg"
    if os.path.exists(image_path):
        with open(image_path, "rb") as image_file:
            admin_client.person_group.add_person_face(
                person_group_id=PERSON_GROUP_ID,
                person_id=person.person_id,
                image_content=image_file.read()
            )
        print(f"Added face from {image_path}")
    else:
        print(f"Warning: {image_path} not found. Please place a sample photo in the folder.")

    # 4. Train the group model
    print("Training PersonGroup model...")
    admin_client.person_group.train(person_group_id=PERSON_GROUP_ID)
    print("Training complete! Your Azure model is ready.")

if __name__ == "__main__":
    setup_and_train()