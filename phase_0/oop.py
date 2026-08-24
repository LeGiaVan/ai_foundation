class Animal:
    def __init__(self, name: str, sound: str, energy: int | None):
        self.name = name
        self.sound = sound
        self.energy: int = 100
    def make_sound(self) -> str:
        return f'Con {self.name} kêu {self.sound}'
    def run(self) -> None:
        self.energy -= 10 
        if self.energy < 0:
            self.energy = 0
        return self.energy
    def sleep(self) -> None:
        self.energy += 30 
        if self.energy >= 100:
            self.energy = 100
        return self.energy
        
class Dog(Animal):
    def __init__(self, name: str, sound: str, breed: str, energy: int = 100):
        super().__init__(name, sound, energy)
        self.breed = breed
    def fetch(self) -> str:
        return f"{self.name} đang tha bóng"

class Cat(Animal):
    def __init__(self, name: str, sound: str, energy: int = 100):
        super().__init__(name, sound, energy)
    def scratch(self) -> str:
        return f"{self.name} đang cào"

animals: list[Animal] = [
    Dog("Lu", "Gâu gâu", "Golden Retriever"),
    Cat("Miu", "Meo meo"),
]

def tire_out_all(animals: list[Animal]) -> None:
    for animal in animals:
        for _ in range(3):
            animal.run()
        print(f"{animal.name}: energy còn {animal.energy}")

tire_out_all(animals)