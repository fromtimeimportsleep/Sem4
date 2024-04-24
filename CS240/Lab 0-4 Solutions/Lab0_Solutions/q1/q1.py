import json
import numpy as np
import matplotlib.pyplot as plt

def inv_transform(distribution: str, num_samples: int, **kwargs) -> list:
    """ populate the 'samples' list from the desired distribution """

    samples = []

    # TODO: first generate random numbers from the uniform distribution

    uniform = np.random.uniform(0, 1, num_samples)

    if (distribution == "exponential"):
        for u in uniform:
            samples.append((-np.log(1-u)/kwargs["lambda"]).round(4))
    
    elif (distribution == "cauchy"):
        for u in uniform:
            samples.append((np.tan(np.pi*(u-0.5))*kwargs["gamma"] + kwargs["peak_x"]).round(4))

    # END TODO
            
    return samples


if __name__ == "__main__":
    np.random.seed(42)

    for distribution in ["cauchy", "exponential"]:
        file_name = "q1_" + distribution + ".json"
        args = json.load(open(file_name, "r"))
        samples = inv_transform(**args)

        with open("q1_output_" + distribution + ".json", "w") as file:
            json.dump(samples, file)

        # TODO: plot and save the histogram to "q1_" + distribution + ".png"

        plt.figure()
        plt.hist(samples, bins=100)
        plt.savefig("q2_" + distribution + ".png")

        # END TODO
