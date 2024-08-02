; Main Controller: This module will coordinate the
; operations of the other modules. It will instruct the
; Automation Module to navigate to a profile, then tell the Screenshot
; Module to take a picture, pass that picture to
; the Decision Module to generate a score and make a decision from that.

; Set-up:
; 100% zoom in resolution settings
; Laptop must be plugged in or else the save/file dialog will lag.
; Greenshot: set output location to Screenshots folder
; 			 set General ---> "capture region" hotkey to f11
;            set Capture --> turn "Show notifications" off
;            ensure output is .png
;            set Output ---> JPEG quality ---> 100%
;			 set Filename ---> screenshot
; Capture2Text: unbind Win + R (under Hotkeys) so we can open run dialog (not needed now but may be needed in the future)
;				turn off Output ---> "show popup window"
; In powershell (run as admin):
; 	Set-ExecutionPolicy Unrestricted -Scope LocalMachine
; 	Unblock-File -Path "...\Github\Operation_Love\beauty_and_BMI_analysis\analyze_beauty.venv\Scripts\Activate.ps1"
;	Unblock-File -Path "...\Github\Operation_Love\beauty_and_BMI_analysis\analyze_BMI.venv\Scripts\Activate.ps1"
; Download C++ built tools


; Warnings:
; Running this while having another one running inside Shadow PC will cause the one outside to crash due to clipboard conflicts.
; If AirDroid, tap become long presses where the context menu pops up, click "Hoykeys" in its menu and click "Switch input method"


#include automation.ahk
#include screenshot.ahk
#include decision.ahk

#SingleInstance
#WinActivateForce

CoordMode "Mouse", "Window"

targeted_index := FileRead("targeted_index.txt") - 1  ; index start at 1. Making it "- 1" just makes loop easier to start at the index indicated in the file.

restart_bumble()
{					
	; Airdroid recent tasks button
	; F-keys like F2 shortcut for this stops working after awhile, use clicks
	MouseClick "left", 1600, 40
	sleep 1000
	
	; Airdroid swipe up button
	; 2 possible locations for some reason
	MouseClick "left", 1770, 40
	sleep 500
	MouseClick "left", 470	, 30
	sleep 2000
	
	; Click Bumble
	MouseClick "left", -350, 380
	
	; Airdroid switch input button
	send "^a"
	
	; Click Bumble
	MouseClick "left", -350, 370
	sleep 9000
}

main(dating_app, root_directory)
{
	; Test
	; restart_bumble()
	
	global targeted_index

	; Total: 1448 cities
	targeted_cities := ["Denver, Colorado", "Glendale, Colorado", "Four Square Mile, Colorado", "Sherrelwood, Colorado", "Aurora, Colorado", "Mountain View, Colorado", "Edgewater, Colorado", "Federal Heights, Colorado", "Cherry Creek, Colorado", "Shaw Heights, Colorado", "Thornton, Colorado", "Centennial, Colorado", "Westminster, Colorado", "Lakewood, Colorado", "Arvada, Colorado", "Highlands Ranch, Colorado", "Stonegate, Colorado", "Sierra Ridge, Colorado", "Boulder, Colorado", "Longmont, Colorado", "Greeley, Colorado", "Fort Collins, Colorado", "Colorado Springs, Colorado", "Cheyenne, Wyoming", "Pueblo, Colorado", "Grand Junction, Colorado", "Santa Fe, New Mexico", "Rio Rancho, New Mexico", "Albuquerque, New Mexico", "Amarillo, Texas", "Provo, Utah", "Orem, Utah", "White City, Utah", "Midvale, Utah", "Taylorsville, Utah", "Salt Lake City, Utah", "West Jordan, Utah", "Kearns, Utah", "West Valley City, Utah", "Ogden, Utah", "Logan, Utah", "Pierre, South Dakota", "Wichita, Kansas", "Oaklawn-Sunview, Kansas", "Lincoln, Nebraska", "Idaho Falls, Idaho", "Billings, Montana", "Lubbock, Texas", "Omaha, Nebraska", "Sioux City, Iowa", "Hartford, South Dakota", "Topeka, Kansas", "Sioux Falls, South Dakota", "Oklahoma City, Oklahoma", "Smith Village, Oklahoma", "St. George, Utah", "Norman, Oklahoma", "Las Cruces, New Mexico", "Tortugas, New Mexico", "Bismarck, North Dakota", "Wichita Falls, Texas", "Olathe, Kansas", "Kansas City, Kansas", "Overland Park, Kansas", "Tulsa, Oklahoma", "El Paso, Texas", "Kansas City, Missouri", "Midland, Texas", "Broken Arrow, Oklahoma", "Odessa, Texas", "Independence, Missouri", "Lee's Summit, Missouri", "Scottsdale, Arizona", "Abilene, Texas", "Mesa, Arizona", "Peoria, Arizona", "Phoenix, Arizona", "Gilbert, Arizona", "San Tan Valley, Arizona", "Tempe, Arizona", "Glendale, Arizona", "Guadalupe, Arizona", "Chandler, Arizona", "Surprise, Arizona", "Helena, Montana", "Des Moines, Iowa", "North Las Vegas, Nevada", "Sunrise Manor, Nevada", "Whitney, Nevada", "Henderson, Nevada", "Winchester, Nevada", "Paradise, Nevada", "Las Vegas, Nevada", "Spring Valley, Nevada", "Enterprise, Nevada", "Tucson, Arizona", "Denton, Texas", "San Angelo, Texas", "Paloma Creek, Texas", "Paloma Creek South, Texas", "Savannah, Texas", "Blue Mound, Texas", "Fargo, North Dakota", "Watauga, Texas", "Fayetteville, Arkansas", "Lewisville, Texas", "Fort Worth, Texas", "Frisco, Texas", "McKinney, Texas", "Hebron, Texas", "Boise, Idaho", "Carrollton, Texas", "Allen, Texas", "Plano, Texas", "Irving, Texas", "Arlington, Texas", "Fort Smith, Arkansas", "Meridian, Idaho", "Springfield, Missouri", "Richardson, Texas", "University Park, Texas", "Grand Prairie, Texas", "Cockrell Hill, Texas", "Garland, Texas", "Nampa, Idaho", "Dallas, Texas", "Mesquite, Texas", "Travis Ranch, Texas", "Columbia, Missouri", "St. Cloud, Minnesota", "Spring Park, Minnesota", "Waterloo, Iowa", "Jefferson City, Missouri", "Richfield, Minnesota", "Minneapolis, Minnesota", "Columbia Heights, Minnesota", "Hilltop, Minnesota", "Lauderdale, Minnesota", "St. Paul, Minnesota", "Rochester, Minnesota", "Landfall, Minnesota", "Cedar Rapids, Iowa", "University Heights, Iowa", "Waco, Texas", "Iowa City, Iowa", "Killeen, Texas", "Avenue B and C, Arizona", "Temple, Texas", "Yuma, Arizona", "Tyler, Texas", "Longview, Texas", "Indio, California", "Round Rock, Texas", "Davenport, Iowa", "Wells Branch, Texas", "Heber, California", "Austin, Texas", "Eau Claire, Wisconsin", "Little Rock, Arkansas", "Victorville, California", "Breckenridge Hills, Missouri", "University City, Missouri", "Clayton, Missouri", "Marlborough, Missouri", "Maplewood, Missouri", "Lakeshire, Missouri", "Pasadena Park, Missouri", "Norwood Court, Missouri", "Beverly Hills, Missouri", "Velda Village Hills, Missouri", "Hillsdale, Missouri", "Northwoods, Missouri", "Wilbur Park, Missouri", "St. George, Missouri", "Flordell Hills, Missouri", "San Bernardino, California", "St. Louis, Missouri", "Glasgow Village, Missouri", "Hemet, California", "Sparks, Nevada", "Moreno Valley, California", "Rialto, California", "Carson City, Nevada", "San Antonio, Texas", "College Station, Texas", "Shreveport, Louisiana", "Fontana, California", "Reno, Nevada", "Menifee, California", "Riverside, California", "Rancho Cucamonga, California", "Jurupa Valley, California", "Duluth, Minnesota", "Murrieta, California", "Temecula, California", "Ontario, California", "Home Gardens, California", "Eastvale, California", "Peoria, Illinois", "Coeur d'Alene, Idaho", "Springfield, Illinois", "Montclair, California", "Grandview, Illinois", "Palmdale, California", "Lancaster, California", "Corona, California", "Coronita, California", "Pomona, California", "Desert View Highlands, California", "Orange Cove, California", "Charter Oak, California", "Escondido, California", "Citrus, California", "Covina, California", "Vincent, California", "West Covina, California", "Visalia, California", "South San Jose Hills, California", "Valinda, California", "South Monrovia Island, California", "Baldwin Park, California", "Vista, California", "Spokane Valley, Washington", "Mayflower Village, California", "La Puente, California", "West Puente Valley, California", "North El Monte, California", "Placentia, California", "Oceanside, California", "Bostonia, California", "Mission Viejo, California", "Avocado Heights, California", "El Monte, California", "Orange, California", "Temple City, California", "Anaheim, California", "Parlier, California", "El Cajon, California", "East San Gabriel, California", "San Pasqual, California", "East Niles, California", "La Habra, California", "Carlsbad, California", "South El Monte, California", "Fullerton, California", "Pasadena, California", "Tustin, California", "San Gabriel, California", "Irvine, California", "Rosemead, California", "Hillcrest, California", "Clovis, California", "Rose Hills, California", "Laguna Woods, California", "Whittier, California", "East Whittier, California", "East Bakersfield, California", "Potomac Park, California", "La Cresta, California", "South San Gabriel, California", "La Crescenta-Montrose, California", "Aliso Viejo, California", "La Mesa, California", "Spokane, Washington", "Alhambra, California", "South Pasadena, California", "Cottonwood, California", "South Whittier, California", "La Mirada, California", "Santa Ana, California", "West Whittier-Los Nietos, California", "Monterey Park, California", "Pico Rivera, California", "Montebello, California", "La Presa, California", "Glendale, California", "San Diego, California", "Buena Park, California", "Lemon Grove, California", "Rexland Acres, California", "Benton Park, California", "Bakersfield, California", "Garden Grove, California", "Mayfair, California", "East Los Angeles, California", "Old Stine, California", "Stanton, California", "Norwalk, California", "La Palma, California", "Madison, Wisconsin", "Cerritos, California", "Fresno, California", "Costa Mesa, California", "Fountain Valley, California", "Downey, California", "Cypress, California", "Bell Gardens, California", "Artesia, California", "Midway City, California", "Burbank, California", "Westminster, California", "Santa Clarita, California", "Chula Vista, California", "Bell, California", "Maywood, California", "Hawaiian Gardens, California", "Cudahy, California", "Bellflower, California", "San Fernando, California", "Huntington Park, California", "South Gate, California", "Huntington Beach, California", "National City, California", "Lakewood, California", "Rossmoor, California", "Paramount, California", "Walnut Park, California", "Lynwood, California", "Florence-Graham, California", "East Rancho Dominguez, California", "Willowbrook, California", "West Hollywood, California", "Compton, California", "Signal Hill, California", "Bloomington, Illinois", "Long Beach, California", "Los Angeles, California", "Rockford, Illinois", "West Rancho Dominguez, California", "Imperial Beach, California", "Beverly Hills, California", "Westmont, California", "View Park-Windsor Hills, California", "West Athens, California", "Inglewood, California", "Gardena, California", "Culver City, California", "The Woodlands, Texas", "Lennox, California", "Hawthorne, California", "Alondra Park, California", "West Carson, California", "Kennewick, Washington", "Del Aire, California", "Lawndale, California", "Santa Monica, California", "Marina del Rey, California", "Torrance, California", "Lomita, California", "Manhattan Beach, California", "Redondo Beach, California", "Hermosa Beach, California", "Simi Valley, California", "Merced, California", "Taft Heights, California", "Thousand Oaks, California", "Casa Conejo, California", "Santa Paula, California", "Mission Bend, Texas", "Monroe, Louisiana", "Houston, Texas", "Sugar Land, Texas", "West University Place, Texas", "Southside Place, Texas", "Boulder Hill, Illinois", "San Buenaventura, California", "Riverbank, California", "Oxnard, California", "Memphis, Tennessee", "Elgin, Illinois", "Aurora, Illinois", "Cloverleaf, Texas", "Champaign, Illinois", "Airport, California", "Channel Islands Beach, California", "Bystrom, California", "Citrus Heights, California", "Modesto, California", "Roseville, California", "Rouse, California", "Bret Harte, California", "South Houston, Texas", "Foothill Farms, California", "Salida, California", "Naperville, Illinois", "Joliet, Illinois", "Antelope, California", "North Highlands, California", "La Riviera, California", "Rosemont, California", "Pearland, Texas", "Hanover Park, Illinois", "Pasadena, Texas", "Arden-Arcade, California", "Newman, California", "August, California", "Elk Grove, California", "Florin, California", "Glendale Heights, Illinois", "Ridgewood, Illinois", "Bonnie Brae, Illinois", "Fruitridge Pocket, California", "Stockton, California", "Parkway, California", "Sacramento, California", "Lemon Hill, California", "Round Lake Beach, Illinois", "Lincoln Village, California", "Bend, Oregon", "Laredo, Texas", "Yuba City, California", "Santa Barbara, California", "Indian Creek, Illinois", "Mount Prospect, Illinois", "League City, Texas", "Chico, California", "La Grange, Illinois", "Stone Park, Illinois", "La Grange Park, Illinois", "Bellwood, Illinois", "Arbury Hills, Illinois", "Appleton, Wisconsin", "West Allis, Wisconsin", "Beaumont, Texas", "Orland Hills, Illinois", "Melrose Park, Illinois", "Brookfield, Illinois", "Maywood, Illinois", "Hickory Hills, Illinois", "Park City, Illinois", "Park Ridge, Illinois", "Summit, Illinois", "University of California-Santa Barbara, California", "Forest Park, Illinois", "Norridge, Illinois", "Elmwood Park, Illinois", "Isla Vista, California", "Harwood Heights, Illinois", "Niles, Illinois", "Berwyn, Illinois", "Milwaukee, Wisconsin", "Chicago Ridge, Illinois", "Kenosha, Wisconsin", "Oak Park, Illinois", "Highwood, Illinois", "Burbank, Illinois", "Oak Lawn, Illinois", "Cicero, Illinois", "Davis, California", "Hometown, Illinois", "Rio Bravo, Texas", "Skokie, Illinois", "Whitefish Bay, Wisconsin", "Evergreen Park, Illinois", "Shorewood, Wisconsin", "Racine, Wisconsin", "Blue Island, Illinois", "Chicago, Illinois", "Hamilton City, California", "Evanston, Illinois", "Calumet Park, Illinois", "Santa Maria, California", "Hollister, California", "Yakima, Washington", "Antioch, California", "Galveston, Texas", "Grover Beach, California", "Greenfield, California", "Port Arthur, Texas", "Vacaville, California", "Guadalupe, California", "University, Mississippi", "Soledad, California", "Green Bay, Wisconsin", "Gilroy, California", "Redding, California", "Suisun City, California", "Fairfield, California", "Concord, California", "Alum Rock, California", "Lake Charles, Louisiana", "Milpitas, California", "San Jose, California", "Pacheco, California", "Contra Costa Centre, California", "Corpus Christi, Texas", "Salinas, California", "Saranap, California", "Fremont, California", "Watsonville, California", "Cambrian Park, California", "Santa Clara, California", "Castroville, California", "Campbell, California", "Hayward, California", "Cherryland, California", "Vallejo, California", "Ashland, California", "Sunnyvale, California", "Evansville, Indiana", "San Lorenzo, California", "San Leandro, California", "Cupertino, California", "Mountain View, California", "Oakland, California", "Piedmont, California", "Capitola, California", "East Palo Alto, California", "Seaside, California", "Kensington, California", "Berkeley, California", "Live Oak, California", "Pleasure Point, California", "Tara Hills, California", "El Cerrito, California", "East Richmond Heights, California", "Montalvin Manor, California", "Albany, California", "Twin Lakes, California", "Alameda, California", "Rollingwood, California", "Emeryville, California", "Stanford, California", "San Pablo, California", "Richmond, California", "North Fair Oaks, California", "Santa Cruz, California", "West Menlo Park, California", "Foster City, California", "Boyes Hot Springs, California", "Pacific Grove, California", "San Carlos, California", "White City, Oregon", "Belmont, California", "San Mateo, California", "Medford, Oregon", "Lafayette, Indiana", "Burlingame, California", "San Francisco, California", "South San Francisco, California", "Millbrae, California", "San Bruno, California", "Alto, California", "Marin City, California", "Daly City, California", "Broadmoor, California", "Santa Rosa, California", "Rohnert Park, California", "Jackson, Mississippi", "Gresham, Oregon", "Clarksville, Tennessee", "Lafayette, Louisiana", "Bloomington, Indiana", "Johnson City, Oregon", "Maywood Park, Oregon", "Eugene, Oregon", "South Bend, Indiana", "Orchards, Washington", "Notre Dame, Indiana", "Vancouver, Washington", "Portland, Oregon", "Minnehaha, Washington", "Las Lomas, Texas", "Gervais, Oregon", "Four Corners, Oregon", "Hayesville, Oregon", "King City, Oregon", "Salem, Oregon", "Muskegon, Michigan", "Cedar Mill, Oregon", "Keizer, Oregon", "Marlene Village, Oregon", "Indianapolis, Indiana", "Oak Hills, Oregon", "Bethany, Oregon", "Aloha, Oregon", "Holland, Michigan", "Elkhart, Indiana", "Hillsboro, Oregon", "Edinburg, Texas", "Cornelius, Oregon", "Baton Rouge, Louisiana", "McAllen, Texas", "Kent, Washington", "Renton, Washington", "Nashville, Tennessee", "Bellevue, Washington", "Bryn Mawr-Skyway, Washington", "Tacoma, Washington", "Kalamazoo, Michigan", "Grand Rapids, Michigan", "White Center, Washington", "Harlingen, Texas", "Bothell East, Washington", "Seattle, Washington", "Mill Creek East, Washington", "Alderwood Manor, Washington", "Everett, Washington", "Mountlake Terrace, Washington", "Lake Stickney, Washington", "North Lynnwood, Washington", "Esperance, Washington", "Marysville, Washington", "Olympia, Washington", "Parkway Village, Kentucky", "Strathmoor Manor, Kentucky", "Strathmoor Village, Kentucky", "Norbourne Estates, Kentucky", "Meadowview Estates, Kentucky", "Louisville, Kentucky", "Bremerton, Washington", "Brownsville, Texas", "Cameron Park, Texas", "Fort Wayne, Indiana", "Blue Ridge Manor, Kentucky", "Sycamore, Kentucky", "Fincastle, Kentucky", "Murfreesboro, Tennessee", "Coldstream, Kentucky", "Worthington Hills, Kentucky", "Mandeville, Louisiana", "Huntsville, Alabama", "Tuscaloosa, Alabama", "Houma, Louisiana", "Bellingham, Washington", "Metairie, Louisiana", "Lansing, Michigan", "New Orleans, Louisiana", "Frankfort, Kentucky", "Terrytown, Louisiana", "Timberlane, Louisiana", "Cheviot, Ohio", "Northbrook, Ohio", "North College Hill, Ohio", "Cincinnati, Ohio", "Elmwood Place, Ohio", "Bellevue, Kentucky", "Birmingham, Alabama", "Norwood, Ohio", "Golf Manor, Ohio", "Deer Park, Ohio", "Madison Place, Ohio", "Dayton, Ohio", "Gulfport, Mississippi", "Lexington, Kentucky", "Saginaw, Michigan", "Ann Arbor, Michigan", "South Lyon, Michigan", "Flint, Michigan", "Toledo, Ohio", "Chattanooga, Tennessee", "Mobile, Alabama", "Keego Harbor, Michigan", "Dearborn Heights, Michigan", "Dearborn, Michigan", "Lincoln Park, Michigan", "Berkley, Michigan", "Oak Park, Michigan", "Detroit, Michigan", "Hazel Park, Michigan", "Hamtramck, Michigan", "Warren, Michigan", "Sterling Heights, Michigan", "Lincoln Village, Ohio", "Eastpointe, Michigan", "Grosse Pointe Park, Michigan", "Harper Woods, Michigan", "Montgomery, Alabama", "Grosse Pointe, Michigan", "Grandview Heights, Ohio", "Columbus, Ohio", "Bexley, Ohio", "Knoxville, Tennessee", "Pensacola, Florida", "Auburn, Alabama", "Kennesaw State University, Georgia", "Lorain, Ohio", "South Fulton, Georgia", "Sandy Springs, Georgia", "Atlanta, Georgia", "Huntington, West Virginia", "Decatur, Georgia", "Clarkston, Georgia", "Lakewood, Ohio", "Gainesville, Georgia", "Columbus, Georgia", "Cleveland, Ohio", "Lakeview Estates, Georgia", "Cleveland Heights, Ohio", "University Heights, Ohio", "Akron, Ohio", "Willowick, Ohio", "Kingsport, Tennessee", "Canton, Ohio", "Johnson City, Tennessee", "Charleston, West Virginia", "Athens, Georgia", "Asheville, North Carolina", "Macon, Georgia", "Panama City, Florida", "Youngstown, Ohio", "Warner Robins, Georgia", "Greenville, South Carolina", "Mauldin, South Carolina", "New Brighton, Pennsylvania", "Rochester, Pennsylvania", "Spartanburg, South Carolina", "Erie, Pennsylvania", "Wesleyville, Pennsylvania", "Avalon, Pennsylvania", "Crafton, Pennsylvania", "Ingram, Pennsylvania", "McKees Rocks, Pennsylvania", "Bellevue, Pennsylvania", "West View, Pennsylvania", "Dormont, Pennsylvania", "Castle Shannon, Pennsylvania", "Hickory, North Carolina", "Mount Oliver, Pennsylvania", "Millvale, Pennsylvania", "Pittsburgh, Pennsylvania", "Brentwood, Pennsylvania", "Sharpsburg, Pennsylvania", "Aspinwall, Pennsylvania", "Swissvale, Pennsylvania", "Edgewood, Pennsylvania", "Wilkinsburg, Pennsylvania", "Charleroi, Pennsylvania", "Verona, Pennsylvania", "Turtle Creek, Pennsylvania", "Arnold, Pennsylvania", "Pitcairn, Pennsylvania", "Brackenridge, Pennsylvania", "Tallahassee, Florida", "Gastonia, North Carolina", "Augusta, Georgia", "Rock Hill, South Carolina", "Charlotte, North Carolina", "Roanoke, Virginia", "Indiana, Pennsylvania", "Concord, North Carolina", "Winston-Salem, North Carolina", "Kenmore, New York", "Buffalo, New York", "Eggertsville, New York", "Dale, Pennsylvania", "High Point, North Carolina", "Columbia, South Carolina", "Greensboro, North Carolina", "Lynchburg, Virginia", "Burlington, North Carolina", "University of Virginia, Virginia", "Charlottesville, Virginia", "State College, Pennsylvania", "Savannah, Georgia", "Rochester, New York", "Saint John Fisher College, New York", "Durham, North Carolina", "Hagerstown, Maryland", "Cary, North Carolina", "Shippensburg University, Pennsylvania", "Gainesville, Florida", "Fayetteville, North Carolina", "Raleigh, North Carolina", "North Charleston, South Carolina", "Stone Ridge, Virginia", "Frederick, Maryland", "Jacksonville, Florida", "Bull Run, Virginia", "Sudley, Virginia", "Charleston, South Carolina", "Loch Lomond, Virginia", "Sterling, Virginia", "Hutchison, Virginia", "McNair, Virginia", "Centreville, Virginia", "Manassas Park, Virginia", "Sugarland Run, Virginia", "Herndon, Virginia", "Fair Oaks, Virginia", "Germantown, Maryland", "Fredericksburg, Virginia", "Lewisburg, Pennsylvania", "Dale City, Virginia", "Burke Centre, Virginia", "Gaithersburg, Maryland", "Montgomery Village, Maryland", "Flower Hill, Maryland", "Shiremanstown, Pennsylvania", "Merrifield, Virginia", "Tysons, Virginia", "McSherrystown, Pennsylvania", "Midway, Pennsylvania", "Occoquan, Virginia", "Idylwood, Virginia", "Harrisburg, Pennsylvania", "West Falls Church, Virginia", "Annandale, Virginia", "Falls Church, Virginia", "North Bethesda, Maryland", "Penbrook, Pennsylvania", "Seven Corners, Virginia", "Aspen Hill, Maryland", "SUNY Oswego, New York", "Ocala, Florida", "Leisure World, Maryland", "Bailey's Crossroads, Virginia", "Enhaut, Pennsylvania", "Somerset, Maryland", "Kingstowne, Virginia", "North Kensington, Maryland", "Friendship Heights Village, Maryland", "Arlington, Virginia", "Wheaton, Maryland", "Chevy Chase Section Three, Maryland", "Martin's Additions, Maryland", "Glenmont, Maryland", "Richmond, Virginia", "Forest Glen, Maryland", "Woodlawn, Virginia", "Alexandria, Virginia", "Rutherford, Pennsylvania", "Ithaca, New York", "Kemp Mill, Maryland", "Silver Spring, Maryland", "Huntington, Virginia", "Hybla Valley, Virginia", "Four Corners, Maryland", "Burnt Mills, Maryland", "Takoma Park, Maryland", "Washington, District of Columbia", "Langley Park, Maryland", "West York, Pennsylvania", "Fairland, Maryland", "Chillum, Maryland", "Adelphi, Maryland", "Mount Rainier, Maryland", "York, Pennsylvania", "Hyattsville, Maryland", "Brentwood, Maryland", "University Park, Maryland", "Glassmanor, Maryland", "North Brentwood, Maryland", "Cottage City, Maryland", "College Park, Maryland", "Hillcrest Heights, Maryland", "Reisterstown, Maryland", "Shamokin, Pennsylvania", "Bladensburg, Maryland", "Spring Hill, Florida", "Jasmine Estates, Florida", "Columbia, Maryland", "Temple Hills, Maryland", "East Riverdale, Maryland", "Coral Hills, Maryland", "Laurel, Maryland", "Suitland, Maryland", "Cedar Heights, Maryland", "Landover Hills, Maryland", "Seat Pleasant, Maryland", "New Carrollton, Maryland", "Landover, Maryland", "Peppermill Village, Maryland", "District Heights, Maryland", "Seabrook, Maryland", "Dallastown, Pennsylvania", "Myrtle Beach, South Carolina", "Glenarden, Maryland", "Waldorf, Maryland", "Mount Carmel, Pennsylvania", "Belleair Bluffs, Florida", "Clearwater, Florida", "Syracuse, New York", "Lebanon, Pennsylvania", "Baltimore, Maryland", "Redington Shores, Florida", "North Redington Beach, Florida", "Parkville, Maryland", "South Highpoint, Florida", "Minersville, Pennsylvania", "Frackville, Pennsylvania", "Leesburg, Florida", "Kenneth City, Florida", "Lancaster, Pennsylvania", "Lealman, Florida", "Egypt Lake-Leto, Florida", "South Pasadena, Florida", "Bear Creek, Florida", "Mahanoy City, Pennsylvania", "University, Florida", "Annapolis, Maryland", "Binghamton University, New York", "St. Petersburg, Florida", "North Beach, Maryland", "Tampa, Florida", "Binghamton, New York", "Plymouth, Pennsylvania", "Kingston, Pennsylvania", "Wilkes-Barre, Pennsylvania", "Freeland, Pennsylvania", "West Lawn, Pennsylvania", "Lincoln Park, Pennsylvania", "Brandon, Florida", "Shillington, Pennsylvania", "West Pittston, Pennsylvania", "West Reading, Pennsylvania", "Greenville, North Carolina", "Reading, Pennsylvania", "Laureldale, Pennsylvania", "Mount Penn, Pennsylvania", "Riverview, Florida", "South Bradenton, Florida", "Scranton, Pennsylvania", "West Samoset, Florida", "Bayshore Gardens, Florida", "Kutztown University, Pennsylvania", "Wilmington, North Carolina", "Lakeland, Florida", "Coatesville, Pennsylvania", "Deltona, Florida", "Pine Hills, Florida", "Daytona Beach Shores, Florida", "Newport News, Virginia", "Boyertown, Pennsylvania", "Kennett Square, Pennsylvania", "Tangelo Park, Florida", "Oak Ridge, Florida", "Lake Sarasota, Florida", "Jacksonville, North Carolina", "Coplay, Pennsylvania", "Bethel Manor, Virginia", "Sky Lake, Florida", "Utica, New York", "West Chester, Pennsylvania", "Azalea Park, Florida", "Allentown, Pennsylvania", "East Greenville, Pennsylvania", "Orlando, Florida", "Royersford, Pennsylvania", "Winter Haven, Florida", "Phoenixville, Pennsylvania", "Hampton, Virginia", "Elsmere, Delaware", "Kissimmee, Florida", "Fountain Hill, Pennsylvania", "Wilmington, Delaware", "Buenaventura Lakes, Florida", "Penns Grove, New Jersey", "Souderton, Pennsylvania", "Norfolk, Virginia", "Media, Pennsylvania", "Wilson, Pennsylvania", "Bridgeport, Pennsylvania", "Norristown, Pennsylvania", "West Easton, Pennsylvania", "Easton, Pennsylvania", "Parkside, Pennsylvania", "Chesapeake, Virginia", "Chester, Pennsylvania", "Lansdale, Pennsylvania", "Dover, Delaware", "Bryn Mawr, Pennsylvania", "Conshohocken, Pennsylvania", "Woodlyn, Pennsylvania", "North Wales, Pennsylvania", "Morton, Pennsylvania", "Haverford College, Pennsylvania", "Rutledge, Pennsylvania", "Folsom, Pennsylvania", "Ridley Park, Pennsylvania", "Ardmore, Pennsylvania", "Drexel Hill, Pennsylvania", "Prospect Park, Pennsylvania", "Clifton Heights, Pennsylvania", "Norwood, Pennsylvania", "Aldan, Pennsylvania", "Glenolden, Pennsylvania", "Penn Wynne, Pennsylvania", "Narberth, Pennsylvania", "Lansdowne, Pennsylvania", "Collingdale, Pennsylvania", "Folcroft, Pennsylvania", "Ambler, Pennsylvania", "East Lansdowne, Pennsylvania", "Sharon Hill, Pennsylvania", "Millbourne, Pennsylvania", "Darby, Pennsylvania", "Yeadon, Pennsylvania", "Colwyn, Pennsylvania", "Arcadia University, Pennsylvania", "Glenside, Pennsylvania", "Roslyn, Pennsylvania", "Jenkintown, Pennsylvania", "Willow Grove, Pennsylvania", "Hatboro, Pennsylvania", "Philadelphia, Pennsylvania", "Warminster Heights, Pennsylvania", "Oak Valley, New Jersey", "Ivyland, Pennsylvania", "Rockledge, Pennsylvania", "Cheltenham Village, Pennsylvania", "Virginia Beach, Virginia", "Camden, New Jersey", "Woodlynne, New Jersey", "Mount Ephraim, New Jersey", "Oaklyn, New Jersey", "Audubon, New Jersey", "Merchantville, New Jersey", "Westmont, New Jersey", "Glendora, New Jersey", "Trevose, Pennsylvania", "Ellisburg, New Jersey", "Hi-Nella, New Jersey", "Kingston Estates, New Jersey", "Lindenwold, New Jersey", "Penndel, Pennsylvania", "The College of New Jersey, New Jersey", "Morrisville, Pennsylvania", "Trenton, New Jersey", "Dover, New Jersey", "Somerville, New Jersey", "Victory Gardens, New Jersey", "Cape Coral, Florida", "Middletown, New York", "Bound Brook, New Jersey", "South Bound Brook, New Jersey", "Princeton Meadows, New Jersey", "Morristown, New Jersey", "Dunellen, New Jersey", "East Franklin, New Jersey", "Rutgers University-Busch Campus, New Jersey", "North Plainfield, New Jersey", "Lake Hiawatha, New Jersey", "Twin Rivers, New Jersey", "New Brunswick, New Jersey", "Palm Bay, Florida", "Plainfield, New Jersey", "Highland Park, New Jersey", "Pine Manor, Florida", "Fanwood, New Jersey", "Jamesburg, New Jersey", "Metuchen, New Jersey", "South River, New Jersey", "Garwood, New Jersey", "Caldwell, New Jersey", "Menlo Park Terrace, New Jersey", "Iselin, New Jersey", "Vauxhall, New Jersey", "Fords, New Jersey", "Singac, New Jersey", "Kiryas Joel, New York", "Connecticut Farms, New Jersey", "Union, New Jersey", "Rahway, New Jersey", "Hopelawn, New Jersey", "Roselle Park, New Jersey", "Roselle, New Jersey", "South Amboy, New Jersey", "William Paterson University of New Jersey, New Jersey", "Perth Amboy, New Jersey", "Haledon, New Jersey", "Upper Montclair, New Jersey", "East Orange, New Jersey", "Glen Ridge, New Jersey", "Watsessing, New Jersey", "Prospect Park, New Jersey", "Suffern, New York", "Carteret, New Jersey", "Ampere North, New Jersey", "Brookdale, New Jersey", "Hawthorne, New Jersey", "Lehigh Acres, Florida", "Paterson, New Jersey", "Silver Lake, New Jersey", "Schenectady, New York", "Clifton, New Jersey", "Elizabeth, New Jersey", "Newark, New Jersey", "East Newark, New Jersey", "Harrison, New Jersey", "Fair Lawn, New Jersey", "Elmwood Park, New Jersey", "Passaic, New Jersey", "Keyport, New Jersey", "North Arlington, New Jersey", "Kaser, New York", "Garfield, New Jersey", "Monsey, New York", "Wallington, New Jersey", "Staten Island, New York", "Rutherford, New Jersey", "Spring Valley, New York", "Bonita Springs, Florida", "Wood-Ridge, New Jersey", "Lodi, New Jersey", "Mount Ivy, New York", "Bayonne, New Jersey", "Hillcrest, New York", "New Square, New York", "Hasbrouck Heights, New Jersey", "Maywood, New Jersey", "Keansburg, New Jersey", "Hackensack, New Jersey", "River Edge, New Jersey", "Poughkeepsie, New York", "West Haverstraw, New York", "Jersey City, New Jersey", "North Middletown, New Jersey", "Little Ferry, New Jersey", "New Milford, New Jersey", "Bogota, New Jersey", "Albany, New York", "Ridgefield Park, New Jersey", "Union City, New Jersey", "Bergenfield, New Jersey", "Hoboken, New Jersey", "Dumont, New Jersey", "West New York, New Jersey", "Fairview, New Jersey", "Palisades Park, New Jersey", "Guttenberg, New Jersey", "Peekskill, New York", "Leonia, New Jersey", "Cliffside Park, New Jersey", "Englewood, New Jersey", "Red Bank, New Jersey", "Fort Lee, New Jersey", "Edgewater, New Jersey", "Nyack, New York", "Manhattan, New York", "Watervliet, New York", "Mechanicville, New York", "Point Pleasant, New Jersey", "Brooklyn, New York", "Highlands, New Jersey", "New York, New York", "Belmar, New Jersey", "Lake Como, New Jersey", "Asbury Park, New Jersey", "Bradley Beach, New Jersey", "Long Branch, New Jersey", "Ocean Grove, New Jersey", "Yonkers, New York", "Loch Arbour, New Jersey", "Golden Gate, Florida", "Bronx, New York", "Bronxville, New York", "Tuckahoe, New York", "Mount Vernon, New York", "Naples Manor, Florida", "New Rochelle, New York", "White Plains, New York", "Queens, New York", "Larchmont, New York", "Saddle Rock Estates, New York", "Great Neck, New York", "Great Neck Plaza, New York", "University Gardens, New York", "Great Neck Gardens, New York", "Russell Gardens, New York", "Kensington, New York", "Manorhaven, New York", "Inwood, New York", "Thomaston, New York", "Bellerose Terrace, New York", "Port Washington North, New York", "Port Chester, New York", "Bellerose, New York", "Baxter Estates, New York", "Cedarhurst, New York", "South Valley Stream, New York", "Woodmere, New York", "Elmont, New York", "Floral Park, New York", "Byram, Connecticut", "North Valley Stream, New York", "South Floral Park, New York", "Munsey Park, New York", "Valley Stream, New York", "North New Hyde Park, New York", "Manhasset Hills, New York", "New Hyde Park, New York", "Stewart Manor, New York", "Hewlett, New York", "Franklin Square, New York", "Herricks, New York", "Garden City Park, New York", "North Lynbrook, New York", "Malverne, New York", "Lynbrook, New York", "Garden City South, New York", "Albertson, New York", "Malverne Park Oaks, New York", "Williston Park, New York", "East Rockaway, New York", "West Hempstead, New York", "Mineola, New York", "Harbor Isle, New York", "Lakeview, New York", "Long Beach, New York", "Island Park, New York", "Rockville Centre, New York", "Stamford, Connecticut", "Oceanside, New York", "Carle Place, New York", "South Hempstead, New York", "Port St. Lucie, Florida", "Westbury, New York", "Baldwin, New York", "Uniondale, New York", "Danbury, Connecticut", "Burlington, Vermont", "Roosevelt, New York", "New Cassel, New York", "Freeport, New York", "Salisbury, New York", "East Meadow, New York", "North Merrick, New York", "Point Lookout, New York", "Merrick, New York", "North Bellmore, New York", "Hicksville, New York", "Winooski, Vermont", "Bellmore, New York", "Levittown, New York", "North Wantagh, New York", "Seaford, New York", "Plainedge, New York", "North Massapequa, New York", "Massapequa, New York", "Farmingdale, New York", "South Farmingdale, New York", "Massapequa Park, New York", "Huntington Station, New York", "East Massapequa, New York", "North Amityville, New York", "North Lindenhurst, New York", "Copiague, New York", "Wyandanch, New York", "Lindenhurst, New York", "North Babylon, New York", "Bridgeport, Connecticut", "North Bay Shore, New York", "Brentwood, New York", "Bay Shore, New York", "Waterbury, Connecticut", "Stony Brook University, New York", "New Haven, Connecticut", "Patchogue, New York", "Cabana Colony, Florida", "New Britain, Connecticut", "Montpelier, Vermont", "West Palm Beach, Florida", "Lake Belvedere Estates, Florida", "Stacey Street, Florida", "Royal Palm Estates, Florida", "Palm Beach Shores, Florida", "Hartford, Connecticut", "Westgate, Florida", "Greenacres, Florida", "Pine Air, Florida", "Acacia Villas, Florida", "Palm Springs, Florida", "Springfield, Massachusetts", "San Castle, Florida", "South Palm Beach, Florida", "Watergate, Florida", "Briny Breezes, Florida", "Coral Springs, Florida", "Hillsboro Pines, Florida", "Tamarac, Florida", "Sunrise, Florida", "Highland Beach, Florida", "Margate, Florida", "North Lauderdale, Florida", "Deerfield Beach, Florida", "Lauderhill, Florida", "Davie, Florida", "Lauderdale Lakes, Florida", "Pompano Beach, Florida", "Pembroke Pines, Florida", "Stock Island, Florida", "Oakland Park, Florida", "Miramar, Florida", "Roosevelt Gardens, Florida", "Broadview Park, Florida", "Lazy Lake, Florida", "Wilton Manors, Florida", "Sea Ranch Lakes, Florida", "Lauderdale-by-the-Sea, Florida", "Fort Lauderdale, Florida", "Palm Springs North, Florida", "Country Club, Florida", "Hialeah Gardens, Florida", "Miami Lakes, Florida", "Hollywood, Florida", "Miami Gardens, Florida", "Doral, Florida", "Hialeah, Florida", "West Park, Florida", "Tamiami, Florida", "Kendall West, Florida", "Ives Estates, Florida", "The Hammocks, Florida", "Kendale Lakes, Florida", "Fountainebleau, Florida", "Hallandale Beach, Florida", "Golden Glades, Florida", "Virginia Gardens, Florida", "Ojus, Florida", "Westchester, Florida", "Westwood Lakes, Florida", "West Little River, Florida", "North Miami Beach, Florida", "Aventura, Florida", "Pinewood, Florida", "The Crossings, Florida", "Country Walk, Florida", "Golden Beach, Florida", "Gladeview, Florida", "North Miami, Florida", "Brownsville, Florida", "West Miami, Florida", "Coral Terrace, Florida", "Sunny Isles Beach, Florida", "Richmond West, Florida", "Norwich, Connecticut", "Richmond Heights, Florida", "Bay Harbor Islands, Florida", "Bal Harbour, Florida", "South Miami, Florida", "North Bay Village, Florida", "Surfside, Florida", "Miami, Florida", "Palmetto Estates, Florida", "South Miami Heights, Florida", "West Perrine, Florida", "Miami Beach, Florida", "Naranja, Florida", "Leisure City, Florida", "Homestead, Florida", "Key Biscayne, Florida", "Worcester, Massachusetts", "Leominster, Massachusetts", "Concord, New Hampshire", "Nashua, New Hampshire", "Manchester, New Hampshire", "Woonsocket, Rhode Island", "Lowell, Massachusetts", "Providence, Rhode Island", "Central Falls, Rhode Island", "Pawtucket, Rhode Island", "Lawrence, Massachusetts", "Watertown Town, Massachusetts", "Cambridge, Massachusetts", "Medford, Massachusetts", "Somerville, Massachusetts", "Boston, Massachusetts", "Melrose, Massachusetts", "Malden, Massachusetts", "Everett, Massachusetts", "Chelsea, Massachusetts", "Revere, Massachusetts", "Quincy, Massachusetts", "Brockton, Massachusetts", "Lynn, Massachusetts", "Salem, Massachusetts", "New Bedford, Massachusetts", "Portland, Maine", "Juneau, Alaska", "Anchorage, Alaska", "Aguadilla, Puerto Rico", "Caban, Puerto Rico", "Aguada, Puerto Rico", "Anasco, Puerto Rico", "Puerto Real, Puerto Rico", "Rafael Gonzalez, Puerto Rico", "Arecibo, Puerto Rico", "Imbery, Puerto Rico", "Yauco, Puerto Rico", "Manati, Puerto Rico", "Adjuntas, Puerto Rico", "Jayuya, Puerto Rico", "Penuelas, Puerto Rico", "Villas del Sol, Puerto Rico", "Villa Calma, Puerto Rico", "Ponce, Puerto Rico", "Campanilla, Puerto Rico", "Las Gaviotas, Puerto Rico", "Villa de Sabana, Puerto Rico", "Luis Llorens Torres, Puerto Rico", "Sabana Seca, Puerto Rico", "Juana Diaz, Puerto Rico", "Pajaros, Puerto Rico", "Naranjito, Puerto Rico", "Bayamon, Puerto Rico", "Potala Pastillo, Puerto Rico", "Guaynabo, Puerto Rico", "Cano Martin Pena, Puerto Rico", "Coamo, Puerto Rico", "Comerio, Puerto Rico", "San Juan, Puerto Rico", "Aibonito, Puerto Rico", "Carolina, Puerto Rico", "Trujillo Alto, Puerto Rico", "Cidra, Puerto Rico", "Hacienda San Jose, Puerto Rico", "Santa Barbara, Puerto Rico", "Salinas, Puerto Rico", "Loiza, Puerto Rico", "Cayey, Puerto Rico", "Valle Hill, Puerto Rico", "Villa Hugo I, Puerto Rico", "Caguas, Puerto Rico", "Suarez, Puerto Rico", "San Isidro, Puerto Rico", "Parcelas de Navarro, Puerto Rico", "Gurabo, Puerto Rico", "Campo Rico, Puerto Rico", "Rio Grande, Puerto Rico", "Juncos, Puerto Rico", "Luquillo, Puerto Rico", "Arroyo, Puerto Rico", "Naguabo, Puerto Rico", "Kailua, Hawaii", "Kaneohe, Hawaii", "Honolulu, Hawaii", "Helemano, Hawaii", "Aiea, Hawaii", "Halawa, Hawaii", "Mililani Mauka, Hawaii", "Waimalu, Hawaii", "Wahiawa, Hawaii", "Waipio Acres, Hawaii", "Mililani Town, Hawaii", "Waipio, Hawaii", "Schofield Barracks, Hawaii", "Waikele, Hawaii", "Waipahu, Hawaii", "Iroquois Point, Hawaii", "West Loch Estate, Hawaii", "Ewa Beach, Hawaii", "Ewa Gentry, Hawaii", "Ewa Villages, Hawaii", "Ocean Pointe, Hawaii", "Makakilo, Hawaii", "Kapolei, Hawaii", "Maili, Hawaii"]
	targeted_zip_codes := ['80201', '80246', 'Not Found', 'Not Found', '80010', '80212', '80214', 'Not Found', 'Not Found', '80031', 'Not Found', 'Not Found', '80030', 'Not Found', '80001', 'Not Found', '80134', '80134', '80301', '80501', '80631', '80521', '80901', '82001', '81001', '81501', '87501', '87124', '87101', '79101', '84601', '84057', '84094', '84047', 'Not Found', '84101', '84081', '84118', 'Not Found', '84201', '84321', '57501', '67201', '67216', '68501', '83401', '59101', '79401', '68101', '51101', '57033', '66601', '57101', '73100', '73115', 'Not Found', '73019', '88001', '88001', '58501', '76301', '66051', '66101', 'Not Found', '74101', '79901', '64101', '79701', '74011', '79760', '64050', 'Not Found', '85250', '79601', '85201', '85345', '85001', '85233', 'Not Found', '85280', '85301', 'Not Found', '85224', '85374', '59601', '50265', '89030', 'Not Found', 'Not Found', '89002', '88902', '89426', '89030', 'Not Found', 'Not Found', '85701', '76201', '76901', 'Not Found', '75068', 'Not Found', '76131', '58078', 'Not Found', '72701', '75029', '76101', '75034', 'Not Found', '75056', '83701', '75006', '75002', '75023', '75014', '76000', '72901', '83642', '65801', '75080', 'Not Found', '75050', '75211', '75040', '83651', '75065', '75149', '75126', '65201', 'Not Found', '55384', '50701', '65101', 'Not Found', '55400', '55421', '55421', 'Not Found', 'Not Found', '55901', '55128', '52401', '52246', '76701', '52240', '76540', '85364', '76501', '85364', '75701', '75601', '92201', '78664', '52800', '78728', '92249', '73301', '54701', '72099', '92392', 'Not Found', 'Not Found', 'Not Found', 'Not Found', 'Not Found', '63123', '63121', '63121', '63121', '63121', 'Not Found', 'Not Found', '63123', '63123', '63136', '92401', 'Not Found', 'Not Found', '92543', '89431', '92551', '92376', '89701', '78201', '77840', '71101', '92331', '89501', '92584', '92501', '91701', 'Not Found', '55801', '92562', '92589', '91758', 'Not Found', 'Not Found', '61601', 'Not Found', '62701', '91763', '62702', '93550', '93534', '91718', 'Not Found', '91766', '93551', '93646', 'Not Found', '92025', '95610', '91722', 'Not Found', '91790', '93277', 'Not Found', 'Not Found', 'Not Found', '91706', '91909', 'Not Found', 'Not Found', '91744', 'Not Found', 'Not Found', '92670', '92049', 'Not Found', '92690', 'Not Found', '91731', '92613', '91780', '92801', '93648', '92019', '91775', '91107', 'Not Found', '90631', '92008', '91733', '92632', '91030', '92680', '91775', '92602', '91770', '92103', '93611', '90601', 'Not Found', '90601', 'Not Found', 'Not Found', 'Not Found', '92562', '91770', 'Not Found', '92656', '91941', '99201', '91801', '91030', '96022', 'Not Found', '90637', '92701', '90606', '91754', '90660', '90640', 'Not Found', '91201', '92101', '90620', '91945', '93307', '93304', '93301', '92641', '93703', 'Not Found', 'Not Found', '90680', '90650', '90623', '53701', '90703', '93650', '92626', '92708', '90239', '90630', '90202', '90701', '92655', '91501', '92683', '91350', '91909', '90201', '90270', '90716', '90201', '90706', '91340', '90255', '90280', '92605', '91950', '90711', '90720', '90723', 'Not Found', '90262', 'Not Found', '90221', 'Not Found', '90069', '90220', '90755', '61701', '90801', '90001', '61101', 'Not Found', '91932', '90209', 'Not Found', 'Not Found', 'Not Found', '90301', '90247', '90230', 'Not Found', '90304', '90250', 'Not Found', 'Not Found', '99336', 'Not Found', '90260', '90401', '90292', '90501', '90717', '90266', '90277', '90254', '93062', '95340', '93268', '91358', '91320', '93060', 'Not Found', '71201', '77001', '77478', 'Not Found', 'Not Found', 'Not Found', 'Not Found', '95367', '93030', '37501', '60120', '60502', 'Not Found', '61820', 'Not Found', '93035', 'Not Found', '95610', '95350', '95661', '95351', '95358', '77587', 'Not Found', '95368', '60540', '60431', '95843', '95660', 'Not Found', 'Not Found', '77581', '60133', '77501', 'Not Found', '95360', '95205', '95624', 'Not Found', '60139', '60432', 'Not Found', '95820', '95201', '95823', '94203', 'Not Found', 'Not Found', '95207', '97459', '78040', '95991', '93101', '60061', '60056', '77573', '95926', '60525', '60165', '60526', '60104', '60448', '54911', 'Not Found', '77701', 'Not Found', '60160', '60513', '60153', '60457', 'Not Found', '60068', '60501', 'Not Found', '60130', 'Not Found', '60635', '93117', '60706', '60714', '60402', '53172', '60415', '53140', '60301', '60040', '60459', '60453', '60650', '95616', '60456', '78046', '60076', 'Not Found', '60805', 'Not Found', '53401', '60406', '60064', '95951', '60201', 'Not Found', '93454', '95023', '98901', '94509', '77550', '93433', '93927', '77640', '95687', '93434', '38677', '93960', '54301', '95020', '96001', '94585', '94533', '94518', '95127', '70601', '95035', '95101', 'Not Found', 'Not Found', '78401', '93901', '94595', '94536', '95076', '95124', '95050', '95012', '95008', '94540', 'Not Found', '94589', 'Not Found', '94085', '47701', '94580', '94577', '95014', '94035', '94601', '94620', '95010', 'Not Found', '93955', 'Not Found', '94701', '95953', 'Not Found', 'Not Found', '94530', 'Not Found', '94806', '94706', '93517', '94501', '94806', '94608', '94305', '94806', '94801', 'Not Found', '95060', '94025', 'Not Found', '95416', '93950', '94070', '97503', '94002', '94401', '97501', '47901', '94010', '94080', '94080', '94030', '94066', '92275', '94965', '94013', '94015', '95401', '94927', '39200', '97030', '37040', '70501', '47401', 'Not Found', '97220', '97401', '46601', '99027', '46556', '98660', '97201', '98661', '78582', '97026', 'Not Found', '97305', '97224', '97301', '49440', 'Not Found', '97307', '97006', '46201', 'Not Found', 'Not Found', 'Not Found', '49422', '46514', '97123', '78539', '97113', '70801', '78501', '98030', '98055', '37201', '98004', '98178', '98401', '49001', '49501', 'Not Found', '78550', 'Not Found', '98060', 'Not Found', 'Not Found', '98201', '98043', 'Not Found', 'Not Found', '6450', '98270', '98501', '40217', '40205', '40205', '40207', '40220', '40201', '98310', '78520', '78526', '46801', '40223', '40223', '40241', '37127', 'Not Found', '40245', '70448', '35801', '35401', '70360', '98225', '70001', '48823', '70112', '40601', 'Not Found', '70056', '45211', 'Not Found', 'Not Found', '45201', '45216', '41073', '35201', 'Not Found', '45237', '45236', 'Not Found', '45401', '39500', '40501', '48601', '48103', '48178', '48501', '43601', '37401', '36601', '48320', '48125', '48120', '48146', '48072', '48237', '48201', '48030', '48212', '48088', '48310', '43228', '48021', 'Not Found', '48225', '36101', '48230', 'Not Found', '43085', '43209', '37901', '32501', '36830', 'Not Found', '44052', 'Not Found', 'Not Found', '30301', '25701', '30030', '30021', '44107', '30501', '31900', '44101', '30012', 'Not Found', 'Not Found', '44301', 'Not Found', '37660', '44701', '37601', '25301', '30601', '28801', '31201', '32401', '44501', '31088', '29601', '29662', '15066', '15074', '29301', '16501', '16510', '15202', '15205', '15205', '15136', '15202', 'Not Found', 'Not Found', 'Not Found', '28601', '15210', 'Not Found', '15112', 'Not Found', 'Not Found', '15215', 'Not Found', 'Not Found', 'Not Found', '15022', '15147', '15145', '15068', '15140', '15014', '32301', '28051', '30901', '29730', '28201', '24000', '15701', '28025', 'Not Found', '14217', '14201', '14226', '15056', '27260', '29169', '27395', '24501', '27215', 'Not Found', '22901', '16801', '31401', '14445', '14618', '27701', '21740', '27511', '17257', '32601', '28301', '27601', '29405', '20105', '20678', '32099', '20109', '20109', '29401', 'Not Found', '20163', '20166', '20171', '20120', '20111', 'Not Found', '20170', '22032', '20874', '22401', '17837', 'Not Found', '22015', '20877', '20886', '20879', '17011', '22081', 'Not Found', '17344', '15060', '22125', 'Not Found', '17101', 'Not Found', '22003', '22040', 'Not Found', 'Not Found', 'Not Found', 'Not Found', '13126', '34470', '20906', 'Not Found', '17113', 'Not Found', '22315', 'Not Found', '20815', '22201', 'Not Found', '20815', '20815', 'Not Found', '23173', 'Not Found', '24381', '22301', 'Not Found', '14850', 'Not Found', '20900', 'Not Found', 'Not Found', '20901', '20901', '20912', 'Not Found', 'Not Found', '17404', 'Not Found', 'Not Found', 'Not Found', '20712', '17370', '20780', '20722', '20782', '20745', '20722', '20722', '20740', 'Not Found', '21136', '17872', '20710', '34606', 'Not Found', '21044', '20748', 'Not Found', '20743', '20707', '20746', '20743', '20784', 'Not Found', 'Not Found', '20784', '20743', '20747', '20706', '17313', '29572', 'Not Found', '20601', '17851', 'Not Found', '33755', '13057', '15783', '21201', 'Not Found', 'Not Found', '21234', '33762', '17954', '17931', '34748', '33709', '17601', 'Not Found', '33614', '33707', '33707', '17948', 'Not Found', '20701', 'Not Found', 'Not Found', '20714', '33601', '13901', '18651', '18704', 'Not Found', '18224', '19609', '19609', '33508', '19607', '18643', '19611', '27833', '19601', 'Not Found', '19606', '33568', 'Not Found', '18501', 'Not Found', 'Not Found', '19530', '28401', '33801', '19320', '32725', 'Not Found', '32118', '23600', '19512', '19348', '32819', 'Not Found', '34241', '28540', '18037', '23665', '32809', '13501', '19380', 'Not Found', '18101', '18041', '32801', '19468', '33880', '19460', '23630', 'Not Found', '34741', '18015', '19801', '34743', '08069', '18964', '23500', '19063', 'Not Found', '19405', '19401', '18042', '18040', 'Not Found', '23320', '15074', '19446', '19901', '19010', '19428', '19094', '19454', '19070', '19041', 'Not Found', '19033', '19078', '19003', '19026', '19076', '19018', '19074', 'Not Found', '19036', 'Not Found', '19072', '19050', 'Not Found', '19032', '19002', '19050', '19079', '19082', '19023', 'Not Found', 'Not Found', '19038', '19038', 'Not Found', '19046', '19090', '19040', '17959', '18974', '08090', '18974', '19046', 'Not Found', '23450', '08100', 'Not Found', '08059', '08107', '08106', '08109', 'Not Found', '08029', '19053', '08034', '08083', '08034', 'Not Found', '19047', '08618', '19067', '08601', '07801', '08876', 'Not Found', '33904', '10940', '08805', '08880', 'Not Found', '07960', '08812', '08873', '08854', 'Not Found', '07034', '08520', '08901', '32905', '07060', '08904', '33907', '07023', '08831', '08840', '08877', '07027', '07006', '08840', '08830', '07088', '08863', '07424', 'Not Found', '07083', '07083', '07065', '08861', '07204', '07203', '08878', '07470', '08861', '07508', '07043', '07017', '07028', '07003', '07508', '10901', '07008', 'Not Found', '07003', '07506', '33936', '07501', '07109', '12301', '07011', '07201', '07101', 'Not Found', '07029', '07410', '07407', '07055', '07735', '07031', '10952', '07026', '10952', '07057', '10301', '07070', '10977', '33923', '07075', '07644', 'Not Found', '07002', '10977', '10977', '07604', '07607', '07734', '07601', '07661', '12600', '10993', '07097', 'Not Found', '07643', '07646', '07603', '12201', '07660', '07087', '07621', '07030', '07628', '07093', '07022', '07650', '07093', '10537', '07605', '07010', '07631', '07701', '07024', '07020', '10960', 'Not Found', '12189', '12118', '08742', '11201', '07716', '10000', '07715', '07719', '07712', '07720', '07740', '07756', '10701', 'Not Found', '34116', '10400', '10708', '10707', '10550', '34113', '10801', '10601', '11427', '10538', '11021', '11020', '11021', 'Not Found', 'Not Found', '11021', 'Not Found', '11050', '11096', 'Not Found', 'Not Found', '11050', '10573', '11426', '11050', '11516', '11581', '11598', '11003', '11001', '06830', 'Not Found', 'Not Found', '11030', '11580', 'Not Found', 'Not Found', '11040', 'Not Found', '11557', '11010', '11040', '11040', 'Not Found', '11565', '11563', 'Not Found', '11507', 'Not Found', '11596', '11518', '11552', '11501', '11558', 'Not Found', '11561', '11558', '11570', '06901', '11572', '11514', 'Not Found', 'Not Found', '11568', '10505', '11553', '06810', '05401', '11575', 'Not Found', '11520', '12577', '11554', 'Not Found', '11569', '11566', 'Not Found', '11801', '05404', '11710', '11756', 'Not Found', '11783', 'Not Found', 'Not Found', '11758', '11735', 'Not Found', '11762', '11746', 'Not Found', 'Not Found', 'Not Found', '11726', '11798', '11757', '11703', '06601', '11706', '11717', '11706', '06701', '11794', '06501', '11772', '33410', '06050', '05601', '33401', '33413', '33417', 'Not Found', '33404', '06057', '33409', 'Not Found', '33406', '33415', 'Not Found', '01089', 'Not Found', 'Not Found', 'Not Found', '33435', 'Not Found', '33073', 'Not Found', 'Not Found', '33487', 'Not Found', 'Not Found', '33441', 'Not Found', 'Not Found', 'Not Found', '33060', '33028', '33040', 'Not Found', '32550', '33311', '33317', '33305', 'Not Found', 'Not Found', 'Not Found', '33301', '33015', '33015', 'Not Found', 'Not Found', '33019', 'Not Found', 'Not Found', '33002', 'Not Found', 'Not Found', 'Not Found', '33179', 'Not Found', 'Not Found', 'Not Found', '33009', 'Not Found', 'Not Found', 'Not Found', 'Not Found', 'Not Found', 'Not Found', '33160', 'Not Found', 'Not Found', '33186', 'Not Found', '33160', 'Not Found', '33160', '33142', 'Not Found', 'Not Found', '33160', 'Not Found', '06360', 'Not Found', 'Not Found', '33154', 'Not Found', 'Not Found', 'Not Found', '33101', 'Not Found', 'Not Found', '33157', '33109', '33032', 'Not Found', '33030', '33149', '01601', '01453', '03301', '03060', '03101', '02895', '01850', '02901', '02863', '02860', '01840', 'Not Found', '02138', '02153', '02143', '02101', '02176', '02148', '01054', '02150', '02151', '02169', '02301', '01901', '01355', '02740', '04101', '99801', '99501', '00603', '00603', '00602', '00610', '00765', '00659', 'Not Found', '00617', '00698', 'Not Found', 'Not Found', '00664', '00624', '00949', '00949', 'Not Found', '00949', '00949', '00949', '00795', '00952', '00795', '00953', '00719', 'Not Found', '00795', 'Not Found', 'Not Found', '00769', '00782', 'Not Found', '00705', 'Not Found', 'Not Found', '00739', '00727', '00729', 'Not Found', '00772', '00736', '00729', '00729', 'Not Found', '00772', '00729', '00778', '00778', '00924', '00745', '00777', '00773', 'Not Found', '00718', '96734', '96744', '96801', 'Not Found', '96701', 'Not Found', '96789', 'Not Found', '96786', '96789', '96789', 'Not Found', '96857', 'Not Found', '96797', '96706', '96706', '96706', '96706', '96706', '96706', '96707', '96707', '96792']

	; test
	; msgbox targeted_cities.length
	; msgbox targeted_zip_codes.length
	; exitApp


	super_likes := 0
	first_loop := true
	; Potentially infiite loading
	no_face_detected := false
	
	navigate_to_discover(dating_app)
	
	; like/superlike/dislike test
	; like(dating_app, root_directory)
	; super_like(dating_app, root_directory)
	; dislike(dating_app)
	; exitApp
	
	super_likes := remaining_super_likes(dating_app)
	
	if super_likes == "o" or super_likes == "O"
	{
		super_likes := 0
	}
	
	if dating_app == "photofeeler"
	{
		super_likes := 2000000000
	}
	
	sleep 500
	
	profile_count := 1
	every_x_profile := 10
	
	while true
	{	
		start:
		
		; Refresh Page Logics
		if dating_app == "tinder"
		{		
			; Bad Gateway or out of profiles or "Aw, snap! Something went wrong..." or every x profiles
			if InStr(ocr(900, 135, 964, 173, 100), "Bad", 0) or InStr(ocr(1058, 711, 1108, 728, 100), "unable", 0) or InStr(ocr(662, 440, 701, 466, 100), "Aw", 0) or Mod(profile_count, every_x_profile) == 0
			{
				profile_count += 1
			
				; Click refresh
				mouseClick "left", 100, 67			 
				sleep 15000
				goto("start")
			}
			
			; Click away "____ likes you"
			mouseClick "left", 1300, 242
			sleep 500
			
			; Click away pride stickers if no "____ likes you"
			mouseClick "left", 1157, 245
			sleep 500
			
			
			; Reset image to position 1
			mouseClick "left", 989, 519
			sleep 500
			
			mouseClick "left", 989, 519
			sleep 500
			
			profile_count += 1
		}
		else if dating_app == "bumble"
		{
			; no_face_detected
			if no_face_detected
			{							
				restart_bumble()
				no_face_detected := false
			}
		
			; Click away any popups including "Second time's a charm" compliment suggestion and "It's a match!"
			mouseClick "left", 701, 71
			sleep 1000	
			; Click discover to close side bar if above popup didn't happen
			MouseClick "left", 962, 1042	
			sleep 1500
			
			; To detect when you run out of people ("Adjust your filters") --> change location
			; if true  ; test
			if first_loop or InStr(ocr(831, 783, 932, 828, 100), "Adjust", 0)
			{
				first_loop := false			
			
				; Click profile
				MouseClick "left", 718, 1041
				sleep 7000
				
				; Click travel mode
				MouseClick "left", 893, 295
				sleep 2000
				
				; If glitch happens where travel mode cannot be clicked
				if InStr(ocr(780, 473, 899, 511, 100), "Spotlight", 0)
				{
					restart_bumble()
					goto("start")
				}
				
				targeted_index += 1
				
				; Restart from beginning if at the end
				if targeted_index > 1448
				{
					targeted_index := 1
				}
				
				send targeted_cities[targeted_index]				
				sleep 5000	
				; Fix for unable to click
				send "{enter}"
				sleep 500
				
				; Click the first city in the list
				MouseClick "left", 957, 380		
				sleep 19000	
				
				; Every 12 hours, there is a glitch that pulls down the notifications screen. Pull it back up
				; Airdroid swipe up button
				; 2 possible locations for the airdroid swipe up button for some reason
				MouseClick "left", 1770, 40
				sleep 500
				MouseClick "left", 470	, 30
				sleep 2000
				
				; revert back to normal coord system
				winactivate "AirDroid"
				
				; Click confirmation popup
				MouseClick "left", 960, 980		
				sleep 2000
				
				; Reset, restart Bumble and redo search if it stalls
				if InStr(ocr(1145, 85, 1253, 127, 500), "Cancel", 0)
				{
					; Location is in database
					if !InStr(ocr(938, 941, 1068, 973, 500), "database", 0)
					{
						; Go back one so when it runs again, it will add one and redo the search
						targeted_index -= 1
						
					}
					
					first_loop := true
					
					restart_bumble()
				}
					
				
				goto("start")
			}
		}
		else if dating_app == "hinge"
		{			
			if no_face_detected
			{					
				; Click refresh
				
				no_face_detected := false
			}
		}
		else if dating_app == "okcupid"
		{
		
			; need to refresh everytime because okcupid has unskippable popups like (you recevied a like) that requires scrolling down and clicking
			; every x profiles
			; no_face_detected
			if no_face_detected
			{					
				; Click refresh
				mouseClick "left", 102, 71		 
				sleep 10000
				
				no_face_detected := false
			}
			
		
			; Click out of skippable popups
			MouseClick "left", 168, 514
			sleep 500
			
			; To detect when you run out of people --> change location
			; if true  ; test
			if first_loop or InStr(ocr(754, 609, 822, 639, 100), "empty", 0)
			{
				first_loop := false			
			
				; Click profile
				MouseClick "left", 938, 161
				sleep 4000
				
				; Click settings
				MouseClick "left", 911, 486
				sleep 5000
				
				loop 11
				{
					send "{tab}"	
					sleep 500
				}
				
				send "{enter}"				
				sleep 500
				
				loop 3
				{
					send "{tab}"	
					sleep 100
				}
								
				targeted_index += 1							
				
				; Restart from beginning if at the end
				if targeted_index > 1448
				{
					targeted_index := 1
				}
				
				targeted_zip_code := targeted_zip_codes[targeted_index]
				
				while targeted_zip_code == 'Not Found'
				{
					targeted_index += 1									
					
					; Restart from beginning if at the end
					if targeted_index > 1448
					{
						targeted_index := 1
					}
					
					targeted_zip_code := targeted_zip_codes[targeted_index]
				}
				
				send targeted_zip_code	
				send "{tab}"
				send "{enter}"
				sleep 2000
				
				; Go back to discover
				MouseClick "left", 85, 159	
				sleep 2000
				
				goto("start")
			}
			
			; Click out of profile zoom-ins from empty text detection
			MouseClick "left", 168, 514
			sleep 500
			
			profile_count += 1
		}
		else if dating_app == "photofeeler" 
		{			
			; Test open_ai_clip
			; if InStr(ocr(387, 338, 424, 361, 100), "Aw", 0)
			
			if first_loop
			{
				; since photofeeler don't use winactivate, this delay gives you time to navigate to the browser.
				sleep 5000
				first_loop := false
			}
			
			if InStr(ocr(769, 145, 925, 245, 100), "Max", 0) or InStr(ocr(387, 338, 424, 361, 100), "Aw", 0)
			{
				; F5 for refresh stops working after repeated uses
				mouseClick "left", 103, 69
				
				sleep 100
				A_Clipboard := ""
				sleep 100
				 
				sleep 4000
				
				goto("start")
			}	
			
			; No more pictures to vote
			if InStr(ocr(660, 205, 772, 251, 100), "Credits", 0)
			{
				send "{LControl down}l"
				send "{LControl up}"
				send "https://www.photofeeler.com/vote/dating"
				send "{Enter}"
				sleep 3000
			}
		}
		
		take_screenshot(dating_app)  
		
		decision := make_decision()		
		
		if decision == "No face detected"
		{
			no_face_detected := true
		}
		else
		{
			no_face_detected := false
		}
		
		; Liking Logic
		if decision == "super_like" && super_likes > 0
		{
			super_like(dating_app, root_directory)
			super_likes -= 1
		}
		else if decision == "super_like" || decision == "like"
		{
			hinge_opener := like(dating_app, root_directory)
		}
		else if decision == "dislike"
		{
			dislike(dating_app)
		}		
		
		move_screenshot(root_directory)
		
		; End of loop procedures
		if dating_app == "tinder"
		{
			; sleep 6000
		}
		else if dating_app == "bumble"
		{
		}
		else if dating_app == "hinge"
		{
			sleep 3000
			; Below cause errors when notification of someone matching with you occurs at the same time as we are clicking off prompt poll popup, it will click the match popup and go to messages page.
			; Click off add a prompt poll popup
			; mouseClick "left", 723, 161
			; sleep 3000
			; mouseClick "left", 828, 567
			; sleep 3000			
			; If no prompt poll, need to remove photo description as a consequence of clicking on it
			; mouseClick "left", 957, 634
		}
		else if dating_app == "okcupid"
		{
		}
		else if dating_app == "photofeeler"
		{
		}
	}
	
}

; "tinder", "bumble", "okcupid", "match", "eharmony", "hinge"

; "1920x1080"
main("tinder", "C:\Users\Bull\Desktop\Github\Operation_Love")

; "1366x768"
; main("okcupid", "C:\Users\Dell\Desktop\GitHub\Operation_Love")

; "1920x1080"
; main("bumble", "C:\Users\Jack.Wu\Documents\GitHub\Operation_Love")

; "1920x1080"
; main("hinge", "C:\Users\super\Desktop\Github\Operation_Love")

; "1366x768"
; main("photofeeler", "C:\Users\LENOVO\Desktop\GitHub\Operation_Love")

f12::
{
	global targeted_index
	
	; May over-subtract, but prevents missing a city
	targeted_index -= 1
	
	FileDelete "targeted_index.txt"
	FileAppend targeted_index, "targeted_index.txt"
	
	exitApp
}