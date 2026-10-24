import 'dart:async';
import 'dart:convert';
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;

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
  String name = '', binType = '', color = '';
  double conf = 0;
  String status = '';
  List<String> guide = [];
  List<Map<String, dynamic>> dets = [];
  final recent = <String>[];

  static const palette = {
    'Green': Color(0xFF2E9E4F), 'Blue': Color(0xFF1E6FD9),
    'Red': Color(0xFFD62F2F), 'Black': Color(0xFF2B2B2B),
  };
  static const icons = {
    'Green': Icons.eco, 'Blue': Icons.recycling,
    'Red': Icons.warning_amber_rounded, 'Black': Icons.delete_outline,
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
      final res = await http.Response.fromStream(await req.send())
        .timeout(const Duration(seconds: 10));
      final j = jsonDecode(res.body);
      final key = (j['item'] ?? 'none').toString();
      recent.add(key);
      if (recent.length > 3) recent.removeAt(0);
      // update only if 2 of the last 3 agree (reduces flicker)
      if (recent.where((c) => c == key).length >= 2) {
        setState(() {
          name = j['name'] ?? '';
          binType = j['bin'] ?? '';
          color = j['color'] ?? '';
          conf = (j['confidence'] as num).toDouble();
          guide = (j['guideline'] as List?)?.map((e) => e.toString()).toList() ?? [];
          dets = ((j['detections'] as List?) ?? [])
              .map((e) => Map<String, dynamic>.from(e)).toList();
        });
      }
      if (mounted && status.isNotEmpty) setState(() => status = '');

    } catch (e) {
      debugPrint('scan error: $e');
      if (mounted) setState(() => status = 'Error: $e');
    }
    busy = false;
  }

  @override
  void dispose() {
    timer?.cancel();
    cam.dispose();
    super.dispose();
  }
  Widget boxesOverlay(double w, double h) {
    double n(dynamic v) => (v as num).toDouble();
    return Stack(children: [
      for (final d in dets)
        if (d['box'] != null)
          Positioned(
            left: n(d['box'][0]) * w,
            top: n(d['box'][1]) * h,
            width: (n(d['box'][2]) - n(d['box'][0])) * w,
            height: (n(d['box'][3]) - n(d['box'][1])) * h,
            child: Container(
              decoration: BoxDecoration(
                border: Border.all(
                    color: palette[d['color']] ?? Colors.white, width: 3)),
              alignment: Alignment.topLeft,
              child: Container(
                color: palette[d['color']] ?? Colors.white,
                padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                child: Text(
                  '${d['name']} ${((d['confidence'] as num) * 100).round()}%',
                  style: const TextStyle(
                    color: Colors.white, fontSize: 12, fontWeight: FontWeight.w600),
                ),
              ),
            ),
          ),
    ]);
  }
  Widget resultCard(Color c) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.fromLTRB(22, 20, 22, 28),
      decoration: BoxDecoration(
        color: c,
        borderRadius: const BorderRadius.vertical(top: Radius.circular(28)),
      ),
      child: Column(mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start, children: [
        Row(children: [
          Icon(icons[color], color: Colors.white, size: 38),
          const SizedBox(width: 12),
          Expanded(
            child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
              Text(name, style: const TextStyle(fontSize: 26, color: Colors.white,
                  fontWeight: FontWeight.bold)),
              Text('${color.toUpperCase()} BIN  •  $binType',
                  style: const TextStyle(fontSize: 15, color: Colors.white70)),
            ]),
          ),
        ]),
        const SizedBox(height: 14),
        Row(children: [
          Expanded(
            child: ClipRRect(
              borderRadius: BorderRadius.circular(6),
              child: LinearProgressIndicator(
                value: conf.clamp(0.0, 1.0), minHeight: 8,
                backgroundColor: Colors.white24, color: Colors.white),
            ),
          ),
          const SizedBox(width: 10),
          Text('${(conf * 100).toStringAsFixed(0)}%',
              style: const TextStyle(color: Colors.white, fontWeight: FontWeight.w600)),
        ]),
        const SizedBox(height: 14),
        const Text('How to dispose',
            style: TextStyle(color: Colors.white, fontSize: 16, fontWeight: FontWeight.w700)),
        const SizedBox(height: 6),
        for (final g in guide)
          Padding(
            padding: const EdgeInsets.only(bottom: 4),
            child: Text('•  $g',
                style: const TextStyle(color: Colors.white, fontSize: 14, height: 1.3)),
          ),
      ]),
    );
  }

  Widget emptyCard() {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.fromLTRB(22, 22, 22, 30),
      decoration: const BoxDecoration(
        color: Color(0xE6101418),
        borderRadius: BorderRadius.vertical(top: Radius.circular(28)),
      ),
      child: const Column(mainAxisSize: MainAxisSize.min, children: [
        Icon(Icons.center_focus_strong, color: Colors.white70, size: 34),
        SizedBox(height: 8),
        Text('Point the camera at a waste item',
            style: TextStyle(color: Colors.white, fontSize: 20, fontWeight: FontWeight.w600)),
        SizedBox(height: 4),
        Text('Hold it steady, one item at a time',
            style: TextStyle(color: Colors.white54, fontSize: 14)),
      ]),
    );
  }

  @override
  Widget build(BuildContext context) {
    if (!ready) {
      return const Scaffold(body: Center(child: CircularProgressIndicator()));
    }
    final found = color.isNotEmpty;
    final c = palette[color] ?? Colors.blueGrey;
    final ps = cam.value.previewSize!;
    return Scaffold(
      backgroundColor: Colors.black,
      body: Stack(children: [
        Positioned.fill(
          child: FittedBox(
            fit: BoxFit.cover,
            child: SizedBox(
                width: ps.height, 
                height: ps.width, 
                 child: Stack(children: [
                  CameraPreview(cam),
                  boxesOverlay(ps.height, ps.width),
                 ]),
            ),
          ),
        ),
        SafeArea(
          child: Align(
            alignment: Alignment.topCenter,
            child: Container(
              margin: const EdgeInsets.only(top: 10),
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
              decoration: BoxDecoration(
                  color: const Color(0x99000000),
                  borderRadius: BorderRadius.circular(20)),
              child: Row(mainAxisSize: MainAxisSize.min, children: [
                const Icon(Icons.recycling, color: Color(0xFF4CD964), size: 20),
                const SizedBox(width: 8),
                Flexible(
                  child: Text(
                    status.isEmpty ? 'WasteWise  •  Live Scanner' : status,
                    maxLines: 2,
                    overflow: TextOverflow.ellipsis,
                    style: const TextStyle(color: Colors.white, fontWeight: FontWeight.w600),
                  ),
                ),
              ]),
            ),
          ),
        ),
        Align(
          alignment: Alignment.bottomCenter,
          child: AnimatedSwitcher(
            duration: const Duration(milliseconds: 250),
            child: found
                ? KeyedSubtree(key: ValueKey(name), child: resultCard(c))
                : KeyedSubtree(key: const ValueKey('empty'), child: emptyCard()),
          ),
        ),
      ]),
    );
  }
}