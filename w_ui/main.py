from functions import prepare_text, search_25
import numpy as np
from rank_bm25 import BM25Okapi
from flask import Flask, request, jsonify
from flask_cors import CORS

documents = [
    'Italian Lasagna - Layers of pasta, meat sauce, and cheese baked to perfection.',
    'Japanese Sushi Rolls - Rice, seaweed, and various fillings like fish, avocado, and cucumber.',
    'Mexican Enchiladas - Tortillas filled with meat or beans, topped with sauce and cheese.',
    'Indian Butter Chicken - Chicken cooked in a creamy tomato sauce with spices.',
    'French Ratatouille - A vegetable stew with eggplant, zucchini, and tomatoes.',
    'Greek Moussaka - Layers of eggplant, meat sauce, and béchamel sauce.',
    'Chinese Sweet and Sour Pork - Pork stir-fried with a tangy sweet and sour sauce.',
    'American BBQ Ribs - Pork ribs slow-cooked and slathered in BBQ sauce.',
    'Thai Green Curry - A spicy curry with coconut milk, green curry paste, and vegetables.',
    'Spanish Paella - A rice dish with seafood, chicken, and saffron.',
    'Moroccan Tagine - A slow-cooked stew with meat, vegetables, and spices.',
    'Lebanese Falafel - Deep-fried balls made from chickpeas, served with pita and tahini.',
    'Korean Bibimbap - A mixed rice dish with vegetables, meat, and a fried egg.',
    'Turkish Kebabs - Grilled skewers of meat, often served with rice or bread.',
    'Vietnamese Pho - A noodle soup with beef or chicken, herbs, and broth.',
    'Brazilian Feijoada - A black bean stew with pork and beef.',
    'Ethiopian Doro Wat - A spicy chicken stew served with injera bread.',
    'Russian Beef Stroganoff - Beef in a creamy sauce with mushrooms, served over noodles.',
    'Caribbean Jerk Chicken - Chicken marinated in a spicy jerk seasoning and grilled.',
    'Middle Eastern Shakshuka - Eggs poached in a spicy tomato and pepper sauce.',
    'Spicy Thai Basil Chicken - A flavorful stir-fry with chicken, basil, and a spicy sauce.',
    'Creamy Garlic Parmesan Pasta - A rich and creamy pasta dish with garlic and Parmesan cheese.',
    'Vegetarian Stuffed Peppers - Bell peppers stuffed with quinoa, black beans, and veggies.',
    'Classic Beef Tacos - Ground beef tacos with all the traditional toppings.',
    'Lemon Herb Grilled Salmon - Salmon fillets marinated in lemon and herbs, then grilled to perfection.',
    'Butternut Squash Soup - A creamy and comforting soup made with roasted butternut squash.',
    'Chicken Alfredo - A creamy pasta dish with chicken and Alfredo sauce.',
    'Mango Avocado Salad - A refreshing salad with mango, avocado, and a lime dressing.',
    'BBQ Pulled Pork Sandwiches - Slow-cooked pulled pork with BBQ sauce, served on buns.',
    'Vegetable Stir-Fry - A quick and healthy stir-fry with a variety of vegetables.',
    'Shrimp Scampi - Shrimp cooked in a garlic butter sauce, served over pasta.',
    'Margherita Pizza - A classic pizza with tomato, mozzarella, and basil.',
    'Chicken Caesar Salad - A hearty salad with grilled chicken, romaine lettuce, and Caesar dressing.',
    'Beef Stroganoff - A creamy beef and mushroom dish served over egg noodles.',
    'Caprese Salad - A simple salad with tomatoes, mozzarella, and basil.',
    'Chicken Tikka Masala - A flavorful Indian dish with chicken in a spiced tomato sauce.',
    'Vegetarian Chili - A hearty chili made with beans, vegetables, and spices.',
    'Garlic Butter Shrimp - Shrimp cooked in a garlic butter sauce, perfect as an appetizer or main dish.',
    'Pesto Pasta - Pasta tossed with a fresh basil pesto sauce.',
    'Chocolate Chip Cookies - Classic cookies with chocolate chips.',
    'Italian Risotto - Creamy rice dish cooked with broth and Parmesan cheese.',
    'Japanese Ramen - Noodle soup with broth, meat, and vegetables.',
    'Mexican Tacos al Pastor - Tacos with marinated pork and pineapple.',
    'Indian Samosas - Fried pastry filled with spiced potatoes and peas.',
    'French Coq au Vin - Chicken braised in red wine with mushrooms and onions.',
    'Greek Spanakopita - Spinach and feta cheese pie in phyllo dough.',
    'Chinese Kung Pao Chicken - Spicy stir-fry with chicken, peanuts, and vegetables.',
    'American Mac and Cheese - Baked pasta with a creamy cheese sauce.',
    'Thai Pad Thai - Stir-fried rice noodles with shrimp, tofu, and peanuts.',
    'Spanish Gazpacho - Cold tomato soup with vegetables.',
    'Moroccan Couscous - Steamed couscous with vegetables and spices.',
    'Lebanese Hummus - Chickpea dip with tahini, garlic, and lemon.',
    'Korean Kimchi - Fermented cabbage with spices.',
    'Turkish Baklava - Sweet pastry with layers of nuts and honey.',
    'Vietnamese Banh Mi - Sandwich with pickled vegetables, meat, and cilantro.',
    'Brazilian Moqueca - Fish stew with coconut milk and tomatoes.',
    'Ethiopian Injera - Spongy flatbread served with various stews.',
    'Russian Borscht - Beet soup with sour cream.',
    'Caribbean Curry Goat - Goat meat cooked in a spicy curry sauce.',
    'Middle Eastern Baba Ganoush - Eggplant dip with tahini and garlic.',
    'Italian Tiramisu - Coffee-flavored dessert with mascarpone cheese.',
    'Japanese Tempura - Battered and deep-fried seafood and vegetables.',
    'Mexican Chiles Rellenos - Stuffed and fried chili peppers.',
    'Indian Paneer Tikka - Grilled paneer cheese with spices.',
    'French Quiche Lorraine - Savory pie with bacon, cheese, and eggs.',
    'Greek Gyros - Meat wrapped in pita with tzatziki sauce.',
    'Chinese Dim Sum - Small steamed or fried dumplings.',
    'American Apple Pie - Classic dessert with apples and cinnamon.',
    'Thai Tom Yum Soup - Hot and sour soup with shrimp.',
    'Spanish Churros - Fried dough pastry with cinnamon sugar.'
]


cleaned_docs = [' '.join(prepare_text(doc)) for doc in documents]

tokenized_docs = [prepare_text(doc) for doc in documents]

model = BM25Okapi(tokenized_docs)
# query = "fried pastry"


# for idx in sorted_results:
#   print("Score: {:.3f} => Document: {}".format(results[idx], documents[idx]))


app = Flask(__name__)
CORS(app)


@app.route('/suggest', methods=['POST'])
def suggest():

    data = request.get_json()

    query = data.get('query')

    results = search_25(query, model)
    sorted_results = np.argsort(results)[::-1]
    normalized_sorted_results = sorted_results.tolist()

    links = []
    for index in normalized_sorted_results[:5]:
        links.append(documents[index])
    response = {
        'suggested': links
    }

    return jsonify(response), 200


if __name__ == '__main__':
    app.run(debug=True)
