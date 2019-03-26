# Car Rental Sample Application

![](https://github.com/pmservice/ai-openscale-tutorials/raw/master/notebooks/images/cars4u_app.png?raw=true)

## Requirements

- python 3
- pip
- python libs: urllib3, watson-machine-learning-client, cfenv, Flask, watson-developer-cloud

User also should have account on Bluemix with active us-south region. In us-south region the following services should be prepared:
- IBM Machine Learning (wml). Please note that `Lite` (free) plan is offered.

Optional services:
- Natural Language Understanding. Please note that `Lite` (free) plan is offered.

## Deployment

### Initial configuration

1. Clone repository and enter cloned project directory:

   ```bash
   git clone https://github.com/pmservice/ai-openscale-tutorials.git
   cd applications/cars-4-you
   ```

2. Update with your services credentials the folowing files:

- `vcaps/wml.vcap` (Machine Learning service credentials).

### Deployment and run on local environment

Run:

```bash
pip install -r requirements.txt
export FLASK_APP=app.py
flask run
```

Application will be available at `127.0.0.1:5000`.

### Deployment and run on IBM Cloud (Bluemix)

1. Modify `manifest.yml` by choosing unique name for your `name` and `route`. 

   ```
   name: <your-name>
   routes:
   - route: <your-name>.mybluemix.net
   ```

2. Run:

   ```bash
   bx api https://api.ng.bluemix.net
   bx login
   bx app push
   ```

Application will be available on bluemix.
