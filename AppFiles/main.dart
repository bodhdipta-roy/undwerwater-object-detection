import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:camera/camera.dart';
import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:http_parser/http_parser.dart';
import 'package:image_picker/image_picker.dart';
import 'package:open_filex/open_filex.dart';
import 'package:path_provider/path_provider.dart';

/// Keep a global list of cameras (used by Live page)
List<CameraDescription> cameras = [];

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  try {
    cameras = await availableCameras();
  } catch (_) {}
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Underwater Detection App',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(useMaterial3: true, colorSchemeSeed: Colors.blue),
      home: const HomePage(),
    );
  }
}

/// ===== API CONFIG =====
/// Make sure this IP:PORT is reachable from your phone/PC.
/// On Android emulator, host machine IP is usually 10.0.2.2:8000 (not LAN IP).
class ApiConfig {
  static const String baseUrl = 'http://192.168.31.168:8000';
  static const String detectImage = '$baseUrl/detect/image';
  static const String detectVideo = '$baseUrl/detect/video';
  static const String health = '$baseUrl/health';
}

/// ===== HOME =====
class HomePage extends StatefulWidget {
  const HomePage({super.key});
  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  bool _checking = false;
  String _status = '';

  @override
  void initState() {
    super.initState();
    _checkHealth();
  }

  Future<void> _checkHealth() async {
    setState(() {
      _checking = true;
      _status = 'Checking API...';
    });
    try {
      final r = await http
          .get(Uri.parse(ApiConfig.health), headers: {'Accept': 'application/json'})
          .timeout(const Duration(seconds: 5));
      if (r.statusCode == 200) {
        setState(() {
          _status = '✓ API Connected';
          _checking = false;
        });
      } else {
        setState(() {
          _status = '✗ API Error: ${r.statusCode}';
          _checking = false;
        });
      }
    } catch (e) {
      setState(() {
        _status = '✗ Cannot connect to API\nCheck IP/port 8000 and firewall';
        _checking = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    final ok = _status.startsWith('✓');

    return Scaffold(
      appBar: AppBar(
        title: const Text('Underwater Detection'),
        backgroundColor: Colors.blue,
        foregroundColor: Colors.white,
        actions: [
          IconButton(onPressed: _checkHealth, icon: const Icon(Icons.refresh), tooltip: 'Check API')
        ],
      ),
      body: Center(
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(20),
          child: Column(
            children: [
              Container(
                padding: const EdgeInsets.all(30),
                decoration: BoxDecoration(color: Colors.blue.shade50, shape: BoxShape.circle),
                child: const Icon(Icons.camera_alt, size: 100, color: Colors.blue),
              ),
              const SizedBox(height: 20),
              const Text('Underwater Object Detection',
                  textAlign: TextAlign.center,
                  style: TextStyle(fontSize: 28, fontWeight: FontWeight.bold)),
              const SizedBox(height: 10),
              Text('Choose your detection method',
                  style: TextStyle(color: Colors.grey.shade600)),
              const SizedBox(height: 30),

              // Live Camera
              SizedBox(
                width: 300,
                height: 56,
                child: ElevatedButton.icon(
                  style: ElevatedButton.styleFrom(backgroundColor: Colors.red, foregroundColor: Colors.white),
                  onPressed: () => Navigator.push(
                    context,
                    MaterialPageRoute(builder: (_) => const LiveDetectionPage()),
                  ),
                  icon: const Icon(Icons.fiber_manual_record),
                  label: const Text('Live Camera'),
                ),
              ),
              const SizedBox(height: 12),

              // Upload Image
              SizedBox(
                width: 300,
                height: 56,
                child: ElevatedButton.icon(
                  style: ElevatedButton.styleFrom(backgroundColor: Colors.green, foregroundColor: Colors.white),
                  onPressed: () => Navigator.push(
                    context,
                    MaterialPageRoute(builder: (_) => const ImageUploadPage()),
                  ),
                  icon: const Icon(Icons.photo_library),
                  label: const Text('Upload Image'),
                ),
              ),
              const SizedBox(height: 12),

              // Upload Video
              SizedBox(
                width: 300,
                height: 56,
                child: ElevatedButton.icon(
                  style: ElevatedButton.styleFrom(backgroundColor: Colors.purple, foregroundColor: Colors.white),
                  onPressed: () => Navigator.push(
                    context,
                    MaterialPageRoute(builder: (_) => const VideoUploadPage()),
                  ),
                  icon: const Icon(Icons.video_library),
                  label: const Text('Upload Video'),
                ),
              ),
              const SizedBox(height: 36),

              // Health card
              Container(
                constraints: const BoxConstraints(maxWidth: 420),
                padding: const EdgeInsets.all(20),
                decoration: BoxDecoration(
                  color: Colors.grey.shade100,
                  borderRadius: BorderRadius.circular(12),
                  border: Border.all(color: Colors.grey.shade300),
                ),
                child: Column(
                  children: [
                    if (_checking) const CircularProgressIndicator() else
                      Icon(ok ? Icons.check_circle : Icons.error,
                          color: ok ? Colors.green : Colors.red, size: 36),
                    const SizedBox(height: 12),
                    Text(_status,
                        textAlign: TextAlign.center,
                        style: TextStyle(
                          color: ok ? Colors.green : Colors.red,
                          fontWeight: FontWeight.w600,
                        )),
                  ],
                ),
              ),
              const SizedBox(height: 12),
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                decoration: BoxDecoration(
                    color: Colors.grey.shade200, borderRadius: BorderRadius.circular(20)),
                child: Text('API: ${ApiConfig.baseUrl}',
                    style: const TextStyle(fontFamily: 'monospace', fontSize: 12)),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

/// ===== IMAGE UPLOAD =====
class ImageUploadPage extends StatefulWidget {
  const ImageUploadPage({super.key});
  @override
  State<ImageUploadPage> createState() => _ImageUploadPageState();
}

class _ImageUploadPageState extends State<ImageUploadPage> {
  final ImagePicker _picker = ImagePicker();
  File? _image;
  bool _busy = false;
  String? _err;
  Map<String, dynamic>? _result;

  Future<void> _pick(ImageSource src) async {
    try {
      final x = await _picker.pickImage(source: src, maxWidth: 1920, maxHeight: 1080, imageQuality: 85);
      if (x != null) {
        setState(() {
          _image = File(x.path);
          _result = null;
          _err = null;
        });
      }
    } catch (e) {
      setState(() => _err = 'Error picking image: $e');
    }
  }

  Future<void> _detect() async {
    if (_image == null) return;
    setState(() {
      _busy = true;
      _err = null;
    });
    try {
      final req = http.MultipartRequest('POST',
          Uri.parse('${ApiConfig.detectImage}?conf=0.5&iou=0.45'));
      req.headers['Accept'] = 'application/json';
      req.files.add(await http.MultipartFile.fromPath(
        'file', _image!.path, contentType: MediaType('image', 'jpeg'),
      ));
      final streamed = await req.send().timeout(const Duration(seconds: 30));
      final res = await http.Response.fromStream(streamed);
      if (res.statusCode == 200) {
        setState(() {
          _result = json.decode(res.body) as Map<String, dynamic>;
          _busy = false;
        });
      } else {
        setState(() {
          _err = 'API Error: ${res.statusCode}';
          _busy = false;
        });
      }
    } catch (e) {
      setState(() {
        _err = 'Error: $e';
        _busy = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    final detections = (_result?['detections'] as List?) ?? [];
    final numFish = _result?['num_fish'] ?? 0;
    final numObjs = _result?['num_objects'] ?? detections.length;

    return Scaffold(
      appBar: AppBar(
          title: const Text('Upload Image'),
          backgroundColor: Colors.green, foregroundColor: Colors.white),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(20),
        child: Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
          Container(
            height: 360,
            decoration: BoxDecoration(
              color: Colors.grey.shade200,
              borderRadius: BorderRadius.circular(12),
              border: Border.all(color: Colors.grey.shade400),
            ),
            child: _image == null
                ? Center(
                child: Text('No image selected', style: TextStyle(color: Colors.grey.shade600)))
                : ClipRRect(
              borderRadius: BorderRadius.circular(10),
              child: Image.file(_image!, fit: BoxFit.contain),
            ),
          ),
          const SizedBox(height: 16),
          Row(children: [
            Expanded(
              child: ElevatedButton.icon(
                onPressed: () => _pick(ImageSource.camera),
                icon: const Icon(Icons.camera_alt),
                label: const Text('Camera'),
              ),
            ),
            const SizedBox(width: 12),
            Expanded(
              child: ElevatedButton.icon(
                onPressed: () => _pick(ImageSource.gallery),
                icon: const Icon(Icons.photo_library),
                label: const Text('Gallery'),
              ),
            ),
          ]),
          const SizedBox(height: 12),
          ElevatedButton.icon(
            onPressed: (_image == null || _busy) ? null : _detect,
            icon: _busy
                ? const SizedBox(
                width: 18, height: 18, child: CircularProgressIndicator(strokeWidth: 2))
                : const Icon(Icons.analytics),
            label: Text(_busy ? 'Processing...' : 'Detect Objects'),
            style: ElevatedButton.styleFrom(backgroundColor: Colors.orange, foregroundColor: Colors.white),
          ),
          if (_err != null) ...[
            const SizedBox(height: 12),
            _errorCard(_err!),
          ],
          if (_result != null) ...[
            const SizedBox(height: 16),
            Text('Detections: $numObjs • Fish: $numFish',
                style: const TextStyle(fontWeight: FontWeight.bold)),
            const SizedBox(height: 8),
            ...detections.map((d) {
              final label = d['label']?.toString() ?? 'unknown';
              final conf = ((d['confidence'] ?? 0.0) as num) * 100;
              final isFish = (d['is_fish'] ?? false) == true;
              return Container(
                margin: const EdgeInsets.only(bottom: 8),
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: isFish ? Colors.blue.shade50 : Colors.orange.shade50,
                  borderRadius: BorderRadius.circular(10),
                  border: Border.all(color: isFish ? Colors.blue : Colors.orange),
                ),
                child: Row(children: [
                  Icon(isFish ? Icons.pets : Icons.block,
                      color: isFish ? Colors.blue : Colors.orange),
                  const SizedBox(width: 10),
                  Expanded(
                      child: Text('$label  •  ${conf.toStringAsFixed(1)}%',
                          style: const TextStyle(fontWeight: FontWeight.w600))),
                ]),
              );
            })
          ],
        ]),
      ),
    );
  }
}

/// ===== VIDEO UPLOAD =====
/// Posts to /detect/video and saves the returned MP4 to a temp file.
/// We avoid gallery_saver to prevent http version conflicts; instead we
/// save locally and offer “Open” (use Share/Open as you like).
class VideoUploadPage extends StatefulWidget {
  const VideoUploadPage({super.key});
  @override
  State<VideoUploadPage> createState() => _VideoUploadPageState();
}

class _VideoUploadPageState extends State<VideoUploadPage> {
  File? _video;
  bool _busy = false;
  String? _err;
  String? _savedPath; // annotated video path after API returns

  Future<void> _pickVideo() async {
    try {
      final result = await FilePicker.platform.pickFiles(type: FileType.video);
      if (result != null && result.files.single.path != null) {
        setState(() {
          _video = File(result.files.single.path!);
          _err = null;
          _savedPath = null;
        });
      }
    } catch (e) {
      setState(() => _err = 'Error picking video: $e');
    }
  }

  Future<void> _sendToApi() async {
    if (_video == null) return;
    setState(() {
      _busy = true;
      _err = null;
      _savedPath = null;
    });

    try {
      final req = http.MultipartRequest('POST', Uri.parse(ApiConfig.detectVideo));
      req.files.add(await http.MultipartFile.fromPath('file', _video!.path,
          contentType: MediaType('video', _video!.path.split('.').last.toLowerCase())));
      final streamed = await req.send().timeout(const Duration(minutes: 5));

      // Expecting FileResponse (video/mp4). Read all bytes.
      final res = await http.Response.fromStream(streamed);
      if (res.statusCode == 200) {
        final bytes = res.bodyBytes;
        final tmpDir = await getTemporaryDirectory();
        final path =
            '${tmpDir.path}/annotated_${DateTime.now().millisecondsSinceEpoch}.mp4';
        final f = File(path);
        await f.writeAsBytes(bytes);
        setState(() {
          _savedPath = path;
          _busy = false;
        });
      } else {
        setState(() {
          _err = 'API Error: ${res.statusCode}';
          _busy = false;
        });
      }
    } catch (e) {
      setState(() {
        _err = 'Error: $e';
        _busy = false;
      });
    }
  }

  Future<void> _openSaved() async {
    if (_savedPath == null) return;
    await OpenFilex.open(_savedPath!);
  }

  @override
  Widget build(BuildContext context) {
    final pickedName = _video?.path.split(Platform.pathSeparator).last;

    return Scaffold(
      appBar: AppBar(
          title: const Text('Upload Video'),
          backgroundColor: Colors.purple, foregroundColor: Colors.white),
      body: Center(
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(20),
          child: Column(mainAxisAlignment: MainAxisAlignment.center, children: [
            Container(
              padding: const EdgeInsets.all(40),
              decoration:
              BoxDecoration(color: Colors.purple.shade50, shape: BoxShape.circle),
              child: Icon(_video != null ? Icons.video_file : Icons.video_library,
                  size: 96, color: Colors.purple),
            ),
            const SizedBox(height: 16),
            Text(_video != null ? 'Video Selected' : 'No Video Selected',
                style: const TextStyle(fontSize: 22, fontWeight: FontWeight.bold)),
            if (pickedName != null) ...[
              const SizedBox(height: 8),
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
                decoration: BoxDecoration(
                    color: Colors.grey.shade200, borderRadius: BorderRadius.circular(8)),
                child: Text(pickedName,
                    maxLines: 2, overflow: TextOverflow.ellipsis,
                    style: TextStyle(color: Colors.grey.shade700)),
              ),
            ],
            const SizedBox(height: 24),
            SizedBox(
              width: 300,
              child: ElevatedButton.icon(
                onPressed: _pickVideo,
                icon: const Icon(Icons.video_library),
                label: const Text('Select Video'),
              ),
            ),
            const SizedBox(height: 12),
            SizedBox(
              width: 300,
              child: ElevatedButton.icon(
                onPressed: (_video == null || _busy) ? null : _sendToApi,
                icon: _busy
                    ? const SizedBox(
                    width: 18, height: 18, child: CircularProgressIndicator(strokeWidth: 2))
                    : const Icon(Icons.cloud_upload),
                label: Text(_busy ? 'Uploading...' : 'Upload & Detect'),
                style: ElevatedButton.styleFrom(
                    backgroundColor: Colors.purple, foregroundColor: Colors.white),
              ),
            ),
            if (_savedPath != null) ...[
              const SizedBox(height: 16),
              Text('Annotated video saved:', style: TextStyle(color: Colors.green.shade700)),
              const SizedBox(height: 6),
              SelectableText(_savedPath!, style: const TextStyle(fontFamily: 'monospace')),
              const SizedBox(height: 10),
              ElevatedButton.icon(
                onPressed: _openSaved,
                icon: const Icon(Icons.play_arrow),
                label: const Text('Open'),
              ),
            ],
            if (_err != null) ...[
              const SizedBox(height: 16),
              _errorCard(_err!),
            ],
          ]),
        ),
      ),
    );
  }
}

/// ===== LIVE DETECTION (Capture a frame and send to /detect/image) =====
class LiveDetectionPage extends StatefulWidget {
  const LiveDetectionPage({super.key});
  @override
  State<LiveDetectionPage> createState() => _LiveDetectionPageState();
}

class _LiveDetectionPageState extends State<LiveDetectionPage> {
  CameraController? _controller;
  Future<void>? _initFuture;
  bool _busy = false;
  String? _err;
  Map<String, dynamic>? _result;
  Timer? _auto;
  bool _autoMode = false;
  int _camIndex = 0;
  bool _flashOn = false;

  @override
  void initState() {
    super.initState();
    _initCamera();
  }

  void _initCamera() {
    if (cameras.isEmpty) {
      setState(() => _err = 'No cameras available');
      return;
    }
    _controller = CameraController(
      cameras[_camIndex],
      ResolutionPreset.high,
      enableAudio: false,
      imageFormatGroup: ImageFormatGroup.jpeg,
    );
    _initFuture = _controller!.initialize().then((_) => setState(() {})).catchError((e) {
      setState(() => _err = 'Camera error: $e');
    });
  }

  @override
  void dispose() {
    _auto?.cancel();
    _controller?.dispose();
    super.dispose();
  }

  Future<void> _switchCam() async {
    if (cameras.length < 2) return;
    setState(() => _camIndex = (_camIndex + 1) % cameras.length);
    await _controller?.dispose();
    _initCamera();
  }

  Future<void> _toggleFlash() async {
    if (_controller == null || !_controller!.value.isInitialized) return;
    try {
      _flashOn = !_flashOn;
      await _controller!.setFlashMode(_flashOn ? FlashMode.torch : FlashMode.off);
      setState(() {});
    } catch (_) {}
  }

  Future<void> _captureAndDetect() async {
    if (_controller == null || !_controller!.value.isInitialized) return;
    if (_busy) return;

    setState(() {
      _busy = true;
      _err = null;
    });
    try {
      final shot = await _controller!.takePicture();
      final f = File(shot.path);

      final req = http.MultipartRequest(
          'POST', Uri.parse('${ApiConfig.detectImage}?conf=0.5&iou=0.45'));
      req.headers['Accept'] = 'application/json';
      req.files.add(await http.MultipartFile.fromPath('file', f.path,
          contentType: MediaType('image', 'jpeg')));
      final streamed = await req.send().timeout(const Duration(seconds: 30));
      final res = await http.Response.fromStream(streamed);

      if (res.statusCode == 200) {
        setState(() {
          _result = json.decode(res.body) as Map<String, dynamic>;
          _busy = false;
        });
      } else {
        setState(() {
          _err = 'API Error: ${res.statusCode}';
          _busy = false;
        });
      }
    } catch (e) {
      setState(() {
        _err = 'Error: $e';
        _busy = false;
      });
    }
  }

  void _toggleAuto() {
    setState(() => _autoMode = !_autoMode);
    if (_autoMode) {
      _auto = Timer.periodic(const Duration(seconds: 3), (_) {
        if (!_busy && _controller != null && _controller!.value.isInitialized) {
          _captureAndDetect();
        }
      });
    } else {
      _auto?.cancel();
    }
  }

  @override
  Widget build(BuildContext context) {
    final detections = (_result?['detections'] as List?) ?? [];
    final numFish = _result?['num_fish'] ?? 0;
    final numObjs = _result?['num_objects'] ?? detections.length;

    return Scaffold(
      appBar: AppBar(
        title: const Text('Live Detection'),
        backgroundColor: Colors.red, foregroundColor: Colors.white,
        actions: [
          Container(
            margin: const EdgeInsets.only(right: 10, top: 8, bottom: 8),
            padding: const EdgeInsets.symmetric(horizontal: 12),
            decoration: BoxDecoration(
              color: _autoMode ? Colors.green : Colors.grey.shade600,
              borderRadius: BorderRadius.circular(20),
            ),
            child: Center(
              child: Text(_autoMode ? 'AUTO ON' : 'AUTO OFF',
                  style: const TextStyle(color: Colors.white, fontWeight: FontWeight.bold)),
            ),
          )
        ],
      ),
      body: Row(
        children: [
          Expanded(flex: 3, child: _cameraView()),
          Expanded(flex: 2, child: _resultsView(detections, numFish, numObjs)),
        ],
      ),
    );
  }

  Widget _cameraView() {
    return Container(
      color: Colors.black,
      child: Column(children: [
        Expanded(
          child: Center(
            child: _controller != null
                ? FutureBuilder(
                future: _initFuture,
                builder: (context, snap) {
                  if (snap.connectionState == ConnectionState.done) {
                    return CameraPreview(_controller!);
                  }
                  return const CircularProgressIndicator(color: Colors.white);
                })
                : (_err != null
                ? Text(_err!, style: const TextStyle(color: Colors.red))
                : const CircularProgressIndicator(color: Colors.white)),
          ),
        ),
        Container(
          padding: const EdgeInsets.all(12),
          color: Colors.grey.shade900,
          child: Row(children: [
            Expanded(
              child: ElevatedButton.icon(
                onPressed: (_controller == null || !_controller!.value.isInitialized || _busy)
                    ? null
                    : _captureAndDetect,
                icon: const Icon(Icons.camera),
                label: const Text('Capture & Detect'),
              ),
            ),
            const SizedBox(width: 8),
            ElevatedButton.icon(
              onPressed: (_controller == null || !_controller!.value.isInitialized)
                  ? null
                  : _toggleAuto,
              icon: Icon(_autoMode ? Icons.stop : Icons.autorenew),
              label: Text(_autoMode ? 'Stop Auto' : 'Auto Mode'),
            ),
            const SizedBox(width: 8),
            if (cameras.length > 1)
              IconButton(
                tooltip: 'Switch Camera',
                onPressed: _switchCam,
                icon: const Icon(Icons.flip_camera_android, color: Colors.white),
              ),
            IconButton(
              tooltip: 'Flash',
              onPressed: _toggleFlash,
              icon: Icon(_flashOn ? Icons.flash_on : Icons.flash_off, color: Colors.white),
            ),
          ]),
        )
      ]),
    );
  }

  Widget _resultsView(List det, int fish, int total) {
    return Container(
      color: Colors.grey.shade100,
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: SingleChildScrollView(
          child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
            const Text('Detection Results',
                style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold, color: Colors.red)),
            const SizedBox(height: 12),
            if (_err != null)
              _errorCard(_err!)
            else if (_result == null)
              Text('No detections yet',
                  style: TextStyle(color: Colors.grey.shade600, fontSize: 16))
            else ...[
                _statRow('Total Objects', '$total', Icons.grid_on),
                _statRow('Fish Detected', '$fish', Icons.pets, Colors.blue),
                const SizedBox(height: 8),
                const Divider(),
                const SizedBox(height: 8),
                const Text('Detected Objects:',
                    style: TextStyle(fontWeight: FontWeight.bold)),
                const SizedBox(height: 8),
                ...det.map((p) {
                  final label = p['label']?.toString() ?? 'unknown';
                  final conf = ((p['confidence'] ?? 0.0) as num) * 100;
                  final isFish = (p['is_fish'] ?? false) == true;
                  return Container(
                    margin: const EdgeInsets.only(bottom: 8),
                    padding: const EdgeInsets.all(10),
                    decoration: BoxDecoration(
                      color: isFish ? Colors.blue.shade50 : Colors.orange.shade50,
                      border: Border.all(color: isFish ? Colors.blue : Colors.orange),
                      borderRadius: BorderRadius.circular(8),
                    ),
                    child: Row(children: [
                      Icon(isFish ? Icons.pets : Icons.block,
                          color: isFish ? Colors.blue : Colors.orange),
                      const SizedBox(width: 8),
                      Expanded(
                          child: Text('$label • ${conf.toStringAsFixed(1)}%')),
                    ]),
                  );
                }),
              ]
          ]),
        ),
      ),
    );
  }

  Widget _statRow(String label, String value, IconData icon, [Color? valueColor]) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4),
      child: Row(children: [
        Icon(icon, size: 18, color: Colors.grey.shade700),
        const SizedBox(width: 8),
        Text(label),
        const Spacer(),
        Text(value,
            style: TextStyle(
                fontWeight: FontWeight.bold, color: valueColor ?? Colors.black)),
      ]),
    );
  }
}

/// Common error card
Widget _errorCard(String msg) {
  return Container(
    padding: const EdgeInsets.all(12),
    decoration: BoxDecoration(
      color: Colors.red.shade50,
      borderRadius: BorderRadius.circular(8),
      border: Border.all(color: Colors.red),
    ),
    child: Row(children: [
      const Icon(Icons.error, color: Colors.red),
      const SizedBox(width: 8),
      Expanded(child: Text(msg, style: const TextStyle(color: Colors.red))),
    ]),
  );
}
