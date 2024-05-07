from flask import Flask, request, render_template
from final_test_caption_generator import testSentences

app = Flask(__name__)


def get_captions(filename):
        try:
            # image = Image.open(filename)
            tokens = testSentences(filename)
            return tokens
            
        except:
            print("ERROR: Couldn't open image! Make sure the image path and extension is correct")
        

@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")

@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
    if request.method == 'POST':
        img = request.files['my_image']
        img_path = "static/" + img.filename	
        img.save(img_path)
        img_tokens = get_captions(img_path)
        # description = generate_desc(model, tokenizer, photo, max_length)
        # description = clearCaption(description)
    return render_template("index.html", prediction = img_tokens, img_path = img_path)

if __name__ =='__main__':
	#app.debug = True
	app.run(debug = True)    