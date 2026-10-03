import 'dart:async';
import 'dart:convert';
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;

// CHANGE THIS to your Mac's IP (use http://localhost:5000 for Chrome)
const apiUrl = 'http://localhost:5050/classify';

late List<CameraDescription> cameras;

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  cameras = await availableCameras();
  runApp(const MaterialApp(debugShowCheckedModeBanner: false, home: Scanner()));
}

class Scanner extends StatefulWidget {
  const Scanner({super.key});
  @override
  State<Scanner> createState() => _ScannerState();
}

class _ScannerState extends State<Scanner> {
  late CameraController cam;
  bool ready = false, busy = false;
  Timer? timer;
  String item = '', binType = '', color = '';
  double conf = 0;
  final recent = <String>[];   // smoothing: last 3 results

  static const palette = {
    'Green': Colors.green, 'Blue': Colors.blue,
    'Red': Colors.red, 'Black': Colors.black87,
  };

  @override
  void initState() {
    super.initState();
    cam = CameraController(cameras.first, ResolutionPreset.medium,
        enableAudio: false);
    cam.initialize().then((_) {
      setState(() => ready = true);
      timer = Timer.periodic(const Duration(milliseconds: 800), (_) => scan());
    });
  }

  Future<void> scan() async {
    if (busy || !cam.value.isInitialized) return;
    busy = true;
    try {
      final shot = await cam.takePicture();
      final req = http.MultipartRequest('POST', Uri.parse(apiUrl))
        ..files.add(http.MultipartFile.fromBytes(
            'image', await shot.readAsBytes(), filename: 'f.jpg'));
      final res = await http.Response.fromStream(await req.send());
      final j = jsonDecode(res.body);
      recent.add(j['color'] ?? 'none');
      if (recent.length > 3) recent.removeAt(0);
      // only update the screen if 2 of the last 3 agree (reduces flicker)
      final agree = recent.where((c) => c == (j['color'] ?? 'none')).length >= 2;
      if (agree) {
        setState(() {
          item = j['item'];
          binType = j['bin'] ?? '';
          color = j['color'] ?? '';
          conf = (j['confidence'] as num).toDouble();
        });
      }
    } catch (e) {
      debugPrint('scan error: $e');
    }
    busy = false;
  }

  @override
  void dispose() {
    timer?.cancel();
    cam.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    if (!ready) return const Scaffold(body: Center(child: CircularProgressIndicator()));
    final c = palette[color] ?? Colors.grey;
    final found = color.isNotEmpty;
    return Scaffold(
      body: Stack(children: [
        Positioned.fill(child: CameraPreview(cam)),
        Align(
          alignment: Alignment.bottomCenter,
          child: Container(
            width: double.infinity,
            padding: const EdgeInsets.all(24),
            color: c.withOpacity(0.92),
            child: Column(mainAxisSize: MainAxisSize.min, children: [
              Text(found ? '${color.toUpperCase()} BIN' : 'Show a waste item',
                  style: const TextStyle(fontSize: 34, color: Colors.white,
                      fontWeight: FontWeight.bold)),
              if (found) Text('$binType • ${(conf * 100).toStringAsFixed(0)}%',
                  style: const TextStyle(fontSize: 18, color: Colors.white)),
            ]),
          ),
        ),
      ]),
    );
  }
}