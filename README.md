# MMSR24

## Hot to run the user-interface

Open the MMSR_Frontend folder in your preferred IDE (I would recommend Webstorm which is free with a student lisence, otherwise VS Code is fine).

Have Node installed (https://nodejs.org/en)

Open a terminal window

Run 'npm install -force', force is needed because of some package conflicts for now unfourtunately

After it sucessfully installed all the packages just run 'ng serve' and navigate to the displayed localhost address (should be http://localhost:4200/).

![alt text](image.png)

## How to run the Flask API

Install all the packages needed in the python scripts. (For the flask server pip install flask and pip install flask_cors should be enough)

Open the flask_server.py in your prefered IDE (I use PyTorch, also free with student lisence), or run the file via a terminal.

If you use Pytorch just use 'Run current file' which lets you hot-reload the API when you make changes to the flask_server.py

![alt text](image-1.png)

If both front- and back-end are running it should look like the screenshots