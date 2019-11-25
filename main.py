import matplotlib.pyplot as plt
import mtcnn
import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument("image_path", help="The path to the image to use.")
parser.add_argument("show_image", help="(True or False) Whether or not to display the image.")

args = parser.parse_args()

if __name__ == "__main__":
	try:
		img = plt.imread(args.image_path)
	except Exception as e:
		print("\t[-] Fatal Error: {}".format(e))
		sys.argv(-1)

	model = mtcnn.MTCNN()
	faces = model.detect_faces(img)

	plt.imshow(img)

	for face in faces:
		confidence = face['confidence']

		ax = plt.gca()

		if confidence >= 0.8:
			x, y, width, height = face['box']
			rect = plt.Rectangle((x, y), width * 1.1, height * 1.1, fill=False, color='red')
			ax.add_patch(rect)

	plt.show()