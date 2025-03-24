#include <opencv2/opencv.hpp>
#include <iostream>
#include <boost/asio.hpp>
#include <vector>

using namespace cv;
using namespace std;
using boost::asio::ip::tcp;

int main() {
    // Abre a câmera
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cout << "Erro ao abrir a câmera!" << endl;
        return -1;
    }

    try {
        // Inicializa o socket cliente
        boost::asio::io_context io_context;
        tcp::resolver resolver(io_context);
        tcp::socket socket(io_context);

        // Resolve o endereço e conecta ao servidor
        auto endpoints = resolver.resolve("192.168.56.1", "12345");
        boost::asio::connect(socket, endpoints);

        Mat frame;
        while (true) {
            cap >> frame;
            if (frame.empty()) break;

            // Codifica o frame como JPEG
            vector<uchar> buf;
            imencode(".jpg", frame, buf);
            uint64_t size = buf.size();  
            // Usa uint64_t para compatibilidade

            // Envia o tamanho da imagem 
            boost::asio::write(socket, boost::asio::buffer(reinterpret_cast<const char*>(&size), sizeof(size)));

            // Envia as imagens
            boost::asio::write(socket, boost::asio::buffer(buf.data(), buf.size()));
            if (waitKey(1) >= 0) break;
        }

        cap.release();
    }
    catch (std::exception& e) {
        cerr << "Erro de socket: " << e.what() << endl;
        return -1;
    }

    return 0;
}
