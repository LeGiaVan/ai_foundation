from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI()

@app.get("/hello/{name}")
def hello(name: str) -> dict: 
    return {"message": f"Xin chào, {name}"}

# cd phase_0
# uvicorn pydantic_fastapi:app --reload

class Student(BaseModel):
    name: str
    age: int = Field(ge=18, le=60)
    email: str = Field(min_length=5)

class StudentInDB(Student):
    id: int
    note: str | None = None

class StudentResponse(StudentInDB):
    msg: str


student_db: dict[int, dict] = {}
next_id = 1

@app.post('/students', response_model=StudentResponse, status_code=201)
def create_student(student: Student) -> StudentResponse:
    global next_id
    new_student = {'id': next_id, **student.model_dump()}
    student_db[next_id] = new_student
    next_id += 1
    return StudentResponse(
        id=next_id-1,
        **student.model_dump(),
        msg=f'Đã thêm thành công {student.name}'
    )

@app.get('/students', response_model=list[StudentInDB])
def get_list_student(min_age: int | None = None, name: str | None = None):
    return [
        s for s in student_db.values()
        if (min_age is None or s['age'] >= min_age)
        and (name is None or name.lower() in s['name'].lower())
    ]

@app.get('/students/{student_id}', response_model=list[StudentInDB])
def get_list_student(id: int | None = None):
    return [
        s for s in student_db.values()
        if s["id"] == id
    ]

